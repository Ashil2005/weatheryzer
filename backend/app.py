from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import date as _date
import os, json, re, logging, time

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- .env ----------
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

# ---------- Optional OpenAI (fallback) ----------
_OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    _OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

app = Flask(__name__)

# ---------- CORS (Netlify/production friendly) ----------
# ALLOWED_ORIGINS can be a comma-separated list, e.g.:
# "https://weatheryzer.earth,https://www.weatheryzer.earth,https://<your>.netlify.app,http://localhost:3000"
_default_allowed = "https://weatheryzer.earth,https://www.weatheryzer.earth,http://localhost:3000"
_ALLOWED_LIST = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", _default_allowed).split(",") if o.strip()]
_ALLOW_ALL = _ALLOWED_LIST == ["*"]

CORS(app,
     resources={r"/*": {"origins": "*" if _ALLOW_ALL else _ALLOWED_LIST}},
     supports_credentials=False)

@app.after_request
def add_cors_headers(resp):
    # Mirror Origin when it matches allowed list; otherwise first allowed (production) or "*"
    origin = request.headers.get("Origin", "")
    if _ALLOW_ALL:
        resp.headers["Access-Control-Allow-Origin"] = "*"
    else:
        allowed = any(origin.startswith(o) for o in _ALLOWED_LIST)
        resp.headers["Access-Control-Allow-Origin"] = origin if (allowed and origin) else _ALLOWED_LIST[0]
    resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Max-Age"] = "3600"
    return resp

# Accept preflight for any path
@app.route("/", methods=["OPTIONS"])
@app.route("/<path:anypath>", methods=["OPTIONS"])
def any_options(anypath=None):
    return ("", 204)

DEFAULT_START = "20000101"

# ---- Tunables for responsiveness ----
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "25"))        # NASA API timeout (seconds)
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "18"))            # single LLM HTTP request timeout
AI_TOTAL_BUDGET = float(os.getenv("AI_TOTAL_BUDGET", "20"))  # total time budget per /ai-advice (seconds)
AI_MODEL_DEFAULT = (os.getenv("OPENROUTER_MODEL") or "google/gemma-2-9b-it:free").strip()

# ----- NASA POWER endpoints -----
POWER_URL_BASE = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M,T2M_MAX,T2M_MIN,WS10M,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"
    "&community=AG&format=JSON"
)
POWER_AOD_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=AOD_550&community=AG&format=JSON"
)

# thresholds
HEAVY_RAIN_MM = 25.0
HEAT_DAY_C = 35.0
SNOW_TEMP_MAX_C = 2.0
SNOW_RAIN_MM = 1.0
FROST_MIN_C = 0.0
HEATWAVE_DAYS = 3
COLD_SPELL_DAYS = 3

_CACHE = {}

# ---------------- helpers ----------------
def parse_any_date(s):
    if s is None: return None
    if isinstance(s, (pd.Timestamp, np.datetime64)): return pd.to_datetime(s)
    if not isinstance(s, str): return None
    s = s.strip()
    if not s: return None
    for f in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try: return pd.to_datetime(s, format=f)
        except Exception: pass
    try: return pd.to_datetime(s)
    except Exception: return None

def to_num(series): return pd.to_numeric(series, errors="coerce")

def safe_float(x):
    try:
        f = float(x)
        return None if not np.isfinite(f) else f
    except Exception:
        return None

def safe_median(series):
    s = to_num(series).dropna()
    return None if s.empty else float(s.median())

def nz(x, default=0.0):
    try:
        f = float(x)
        return default if not np.isfinite(f) else f
    except Exception:
        return default

def _require_keys(body, keys):
    """return missing keys list"""
    return [k for k in keys if k not in body]

# ---------------- fetch ----------------
def fetch_power(lat, lon, start=DEFAULT_START, end=None):
    if end is None:
        end = _date.today().strftime("%Y%m%d")
    key = (round(float(lat), 3), round(float(lon), 3), str(start), str(end))
    if key in _CACHE:
        return _CACHE[key].copy()

    url = f"{POWER_URL_BASE}&start={start}&end={end}&latitude={lat}&longitude={lon}"
    r = requests.get(url, timeout=HTTP_TIMEOUT); r.raise_for_status()
    payload = r.json()
    params = payload["properties"]["parameter"]

    def series(k): return params.get(k, {})

    df = pd.DataFrame({
        "date": list(series("T2M").keys()),
        "t2m": list(series("T2M").values()),
        "t2m_max": list(series("T2M_MAX").values()),
        "t2m_min": list(series("T2M_MIN").values()),
        "ws10m": list(series("WS10M").values()),
        "precip": list(series("PRECTOTCORR").values()),
        "rh2m": list(series("RH2M").values()),
        "allsky": list(series("ALLSKY_SFC_SW_DWN").values()),
        "clrsky": list(series("CLRSKY_SFC_SW_DWN").values()),
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in ["t2m","t2m_max","t2m_min","ws10m","precip","rh2m","allsky","clrsky"]:
        df[c] = to_num(df[c])

    with np.errstate(divide="ignore", invalid="ignore"):
        cloud_frac = 1 - (df["allsky"] / df["clrsky"])
    df["cloud_pct"] = np.clip(cloud_frac, 0, 1) * 100.0

    df["aod550"] = np.nan
    try:
        aod_url = f"{POWER_AOD_URL}&start={start}&end={end}&latitude={lat}&longitude={lon}"
        r2 = requests.get(aod_url, timeout=HTTP_TIMEOUT); r2.raise_for_status()
        p2 = r2.json()["properties"]["parameter"]
        aod = p2.get("AOD_550", {})
        aod_map = {pd.to_datetime(k, format="%Y%m%d"): safe_float(v) for k, v in aod.items()}
        df["aod550"] = df["date"].map(aod_map)
    except Exception:
        pass

    _CACHE[key] = df.copy()
    return df

# ---------------- stats ----------------
def heat_index_celsius(t_c, rh):
    t_f = t_c * 9/5 + 32; R = rh
    HI_f = (-42.379 + 2.04901523*t_f + 10.14333127*R - 0.22475541*t_f*R
            - 0.00683783*t_f*t_f - 0.05481717*R*R + 0.00122874*t_f*t_f*R
            + 0.00085282*t_f*R*R - 0.00000199*t_f*t_f*R*R)
    HI_f = np.where(t_f < 80, t_f, HI_f)
    return (HI_f - 32) * 5/9

def seasonal_subset(df, date_like, window_days=7):
    d = parse_any_date(date_like)
    if d is None or pd.isna(d): return df.iloc[0:0].copy()
    mask = (df["date"].dt.month == d.month) & (abs(df["date"].dt.day - d.day) <= int(window_days))
    return df[mask].copy()

def default_thresholds():
    return {
        "Very Hot": 32.0, "Very Cold": 0.0, "Very Wet": 10.0,
        "Very Windy": 10.0, "Very Uncomfortable": 32.0,
    }

def summarize(values):
    s = pd.Series(values).dropna()
    if s.empty:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "mean": round(float(s.mean()), 1),
        "p10": round(float(np.percentile(s, 10)), 1),
        "p50": round(float(np.percentile(s, 50)), 1),
        "p90": round(float(np.percentile(s, 90)), 1),
        "min": round(float(s.min()), 1),
        "max": round(float(s.max()), 1),
    }

def latest_same_day(df, metric, m, d):
    exact = df[(df["date"].dt.month == m) & (df["date"].dt.day == d)].copy()
    if exact.empty: return None
    row = exact.sort_values("date").iloc[-1]
    val = safe_float(row[metric])
    return None if val is None else {"date": row["date"].strftime("%Y-%m-%d"), "value": round(val, 1)}

def compute_condition_stats(sub, condition, overrides=None, for_date=None, full_df=None):
    th = default_thresholds()
    if overrides and isinstance(overrides, dict):
        for k, v in overrides.items():
            if k in th and isinstance(v, (int, float)): th[k] = float(v)

    cond = (condition or "").strip().lower()
    metric, unit, threshold = "t2m_max", "°C", th["Very Hot"]

    if cond == "very hot":
        metric, unit, threshold = "t2m_max", "°C", th["Very Hot"]; raw = to_num(sub[metric]).to_numpy()
    elif cond == "very cold":
        metric, unit, threshold = "t2m_min", "°C", th["Very Cold"]; raw = to_num(sub[metric]).to_numpy()
    elif cond == "very wet":
        metric, unit, threshold = "precip", "mm/day", th["Very Wet"]; raw = to_num(sub[metric]).to_numpy()
    elif cond == "very windy":
        metric, unit, threshold = "ws10m", "m/s", th["Very Windy"]; raw = to_num(sub[metric]).to_numpy()
    elif cond == "very uncomfortable":
        hi = heat_index_celsius(to_num(sub["t2m"]).to_numpy(), to_num(sub["rh2m"]).to_numpy())
        raw = hi; metric, unit, threshold = "heat_index", "°C", th["Very Uncomfortable"]
    else:
        raw = to_num(sub["t2m_max"]).to_numpy()

    if sub.empty or raw is None or len(raw) == 0:
        return 0.0, metric, threshold, unit, [], {}, None

    finite = np.isfinite(raw)
    vals = raw[finite]
    if vals.size == 0:
        return 0.0, metric, threshold, unit, [], {}, None

    prob = float((vals > threshold).mean())

    dates = sub["date"].to_numpy()
    series = [{"date": pd.to_datetime(d).strftime("%Y-%m-%d"), "value": float(v)}
              for d, v, ok in zip(dates, raw, finite) if ok]

    climo = summarize(vals)
    expected = {"median": climo["p50"], "p10": climo["p10"], "p90": climo["p90"]}

    latest = None
    if for_date and full_df is not None:
        dd = parse_any_date(for_date)
        if dd is not None:
            if metric == "heat_index":
                exact = full_df[(full_df["date"].dt.month == dd.month) & (full_df["date"].dt.day == dd.day)].copy()
                if not exact.empty:
                    hi_exact = heat_index_celsius(to_num(exact["t2m"]).to_numpy(),
                                                  to_num(exact["rh2m"]).to_numpy())
                    exact["hi"] = pd.Series(hi_exact, index=exact.index)
                    valid = exact[exact["hi"].notna()].copy()
                    if not valid.empty:
                        row = valid.sort_values("date").iloc[-1]
                        latest = {"date": row["date"].strftime("%Y-%m-%d"),
                                  "value": round(float(row["hi"]), 1)}
            else:
                latest = latest_same_day(full_df, metric, dd.month, dd.day)

    return prob, metric, threshold, unit, series, expected, latest

def label_for(cond):
    return {
        "Very Hot": "🔥 Very Hot",
        "Very Cold": "❄️ Very Cold",
        "Very Wet": "🌧️ Very Wet",
        "Very Windy": "🌬️ Very Windy",
        "Very Uncomfortable": "🥵 Very Uncomfortable",
    }.get(cond, cond)

def analyze_all(sub, overrides=None, for_date=None, full_df=None):
    conditions = ["Very Hot","Very Cold","Very Wet","Very Windy","Very Uncomfortable"]
    ranked, top_detail, best = [], None, -1.0
    for cond in conditions:
        p, metric, threshold, unit, series, expected, latest = compute_condition_stats(
            sub, cond, overrides=overrides, for_date=for_date, full_df=full_df
        )
        item = {
            "condition": cond, "label": label_for(cond),
            "probability": round(p*100,1), "metric": metric, "threshold": threshold,
            "unit": unit, "samples": len(series), "expected": expected,
            "latest_observed": latest, "series": series,
        }
        ranked.append(item)
        if p > best: best, top_detail = p, item
    ranked.sort(key=lambda x: x["probability"], reverse=True)
    return {"ranked": ranked, "top": top_detail}

def _safe_date(y,m,d):
    try: return pd.Timestamp(year=int(y), month=int(m), day=int(d))
    except Exception: return None

def prob_consecutive_event(df, date_str, metric, cmp, thresh, days=3, center=True):
    d = parse_any_date(date_str)
    if d is None: return None
    years = sorted(df["date"].dt.year.dropna().unique().tolist())
    successes, denom = 0, 0
    for y in years:
        d0 = _safe_date(y, d.month, d.day)
        if d0 is None: continue
        offsets = [-1,0,1] if (center and days==3) else list(range(days))
        dates = [d0 + pd.Timedelta(days=o) for o in offsets]
        group = df[df["date"].isin(dates)].copy()
        if len(group) != len(dates): continue
        vals = to_num(group[metric]).to_numpy()
        if not np.all(np.isfinite(vals)):
            denom += 1
            continue
        if all(cmp(v, thresh) for v in vals): successes += 1
        denom += 1
    return None if denom==0 else round(100.0*successes/denom, 1)

def daily_snapshot(df, date_str, window_days=7):
    d = parse_any_date(date_str)
    sub = seasonal_subset(df, d, window_days)

    expected = {
        "t2m": safe_median(sub["t2m"]) if "t2m" in sub else None,
        "t2m_max": safe_median(sub["t2m_max"]) if "t2m_max" in sub else None,
        "t2m_min": safe_median(sub["t2m_min"]) if "t2m_min" in sub else None,
        "precip": safe_median(sub["precip"]) if "precip" in sub else None,
        "ws10m": safe_median(sub["ws10m"]) if "ws10m" in sub else None,
        "rh2m": safe_median(sub["rh2m"]) if "rh2m" in sub else None,
        "cloud_pct": safe_median(sub["cloud_pct"]) if "cloud_pct" in sub else None,
        "aod550": safe_median(sub["aod550"]) if "aod550" in sub else None,
    }

    def chance_ge(series, val):
        s = to_num(series).dropna()
        return None if s.empty else round(float((s >= val).mean()*100.0), 1)

    def chance_lt(series, val):
        s = to_num(series).dropna()
        return None if s.empty else round(float((s < val).mean()*100.0), 1)

    rain_chance = chance_ge(sub["precip"], 1.0) if "precip" in sub else None
    heavy_rain_chance = chance_ge(sub["precip"], HEAVY_RAIN_MM) if "precip" in sub else None
    frost_chance = chance_lt(sub["t2m_min"], FROST_MIN_C) if "t2m_min" in sub else None
    heat_day_chance = chance_ge(sub["t2m_max"], HEAT_DAY_C) if "t2m_max" in sub else None

    snow_chance = None
    if "t2m_max" in sub and "precip" in sub:
        t = to_num(sub["t2m_max"]); p = to_num(sub["precip"])
        ok = (~t.isna()) & (~p.isna())
        if ok.any():
            snow_chance = round(float(((t[ok] < SNOW_TEMP_MAX_C) & (p[ok] >= SNOW_RAIN_MM)).mean()*100.0), 1)

    heatwave_chance = prob_consecutive_event(
        df, d, metric="t2m_max", cmp=lambda v,t: v>=t, thresh=HEAT_DAY_C, days=HEATWAVE_DAYS, center=True
    )
    cold_spell_chance = prob_consecutive_event(
        df, d, metric="t2m_min", cmp=lambda v,t: v<t, thresh=FROST_MIN_C, days=COLD_SPELL_DAYS, center=True
    )

    last_year = None
    if d is not None:
        last = df[(df["date"].dt.month==d.month) & (df["date"].dt.day==d.day)].sort_values("date")
        if not last.empty:
            r = last.iloc[-1]
            last_year = {
                "date": r["date"].strftime("%Y-%m-%d"),
                "t2m": safe_float(r["t2m"]),
                "t2m_max": safe_float(r["t2m_max"]),
                "t2m_min": safe_float(r["t2m_min"]),
                "precip": safe_float(r["precip"]),
                "ws10m": safe_float(r["ws10m"]),
                "rh2m": safe_float(r["rh2m"]),
                "cloud_pct": safe_float(r["cloud_pct"]),
                "aod550": safe_float(r["aod550"]),
            }

    aod_med = expected.get("aod550")
    aq_cat = None
    if aod_med is not None:
        aq_cat = "Good" if aod_med < 0.1 else ("Moderate" if aod_med < 0.3 else "Poor")

    details = {
        "rain_chance_pct": rain_chance,
        "heavy_rain_chance_pct": heavy_rain_chance,
        "frost_chance_pct": frost_chance,
        "snow_chance_pct": snow_chance,
        "heat_day_chance_pct": heat_day_chance,
        "heatwave_chance_pct": heatwave_chance,
        "cold_spell_chance_pct": cold_spell_chance,
        "air_quality": {
            "aod550_median": None if aod_med is None else round(float(aod_med), 3),
            "category": aq_cat
        },
    }

    return expected, last_year, details

# ---------- Activity-aware scoring ----------
def _classify_task(task: str) -> str:
    t = re.sub(r"\s+", " ", (task or "").lower()).strip()

    gaming = [
        "playstation","ps5","ps4","xbox","nintendo","switch","console",
        "pc","computer","laptop","steam","gaming","video game","videogame",
        "rdr","red dead","gta","valorant","csgo","counter strike","cod","call of duty","pubg","bgmi"
    ]
    indoor_terms = [
        "indoor","inside","at home","home","house","apartment","room","living room","bedroom",
        "office","meeting","conference","class","lecture","study","library","gym","yoga",
        "movie","cinema","theatre","theater","mall","shopping","museum","board game","chess",
        "dance class"
    ]
    outdoor_places = [
        "outside","outdoor","open air","open-air","under the sky","open sky","at the sky","beneath the sky",
        "rooftop","terrace","balcony","street","beach","park","garden","ground","field","lawn",
        "stadium","open ground","playground","courtyard","yard"
    ]
    sport = [
        "football","soccer","cricket","basketball","volleyball","tennis",
        "running","run","jog","marathon","cycle","cycling","bike","biking",
        "hike","trek","golf","surf","swim","skate","skating","practice","training"
    ]
    party_gather = ["picnic","bbq","barbecue","party","festival","wedding","ceremony","gathering","camp","camping"]
    photo = ["photo","photoshoot","shoot","filming","videography","drone"]
    work = ["construction","work","site","install","repair","maintenance"]

    if any(k in t for k in gaming):
        return "indoor_gaming"
    if any(k in t for k in work):
        return "work"

    indoor_hits  = sum(k in t for k in indoor_terms)
    outdoor_hits = sum(k in t for k in outdoor_places) \
                   + sum(k in t for k in sport) \
                   + sum(k in t for k in party_gather) \
                   + sum(k in t for k in photo)

    if re.search(r"(under|open|beneath|at)\s+the\s+sky", t):
        outdoor_hits += 4

    if outdoor_hits > indoor_hits:
        if any(k in t for k in sport): return "outdoor_sport"
        if any(k in t for k in photo): return "photo"
        if "dance" in t or any(k in t for k in party_gather): return "outdoor_gathering"
        return "generic_outdoor"

    if "dance" in t and not any(p in t for p in outdoor_places):
        return "indoor"
    return "indoor"

def _bound(x, lo=0.0, hi=100.0): return float(np.clip(float(x), lo, hi))

def _base_reasons_precautions(act, tmax, tmin, wind, rain, heavy, heatday, frost, aq):
    reasons = []
    if tmax is not None: reasons.append(f"Typical max ~{round(tmax,1)}°C")
    if tmin is not None: reasons.append(f"Typical min ~{round(tmin,1)}°C")
    if rain is not None and act != "indoor_gaming":
        reasons.append(f"Rain chance {round(rain,1)}% (heavy {round((heavy or 0),1)}%)")
    if wind is not None and act != "indoor_gaming":
        reasons.append(f"Wind ~{round(wind,1)} m/s")
    if heatday and act != "indoor_gaming": reasons.append(f"Hot-day chance {heatday}%")
    if frost and act != "indoor_gaming": reasons.append(f"Frost chance {frost}%")
    if aq and act != "indoor_gaming": reasons.append(f"Air quality: {aq}")
    reasons = [r for r in reasons if r][:4]

    prec = []
    if act.startswith("outdoor") and (rain or 0) > 30: prec.append("Have rain cover/backup.")
    if act.startswith("outdoor") and (heavy or 0) > 20: prec.append("Avoid low-lying/poor drainage areas.")
    if act.startswith("outdoor") and (tmax or 0) > 32: prec.append("Prefer morning/evening; hydrate & shade.")
    if act.startswith("outdoor") and (tmin or 99) < 10: prec.append("Layer up; gloves if early.")
    if act.startswith("outdoor") and (wind or 0) > 8: prec.append("Secure equipment; avoid loose canopies.")
    if act == "indoor_gaming":
        prec.append("Charge controllers; ensure stable power/internet.")
        if (heavy or 0) > 30 or (wind or 0) > 12: prec.append("Have offline option in case of outages.")
    if aq == "Poor" and act.startswith("outdoor"): prec.append("Limit strenuous activity; masks for sensitive.")
    return reasons, prec[:4]

def compute_activity_assessment(task, expected, details, ranked):
    act = _classify_task(task)
    tmax = expected.get("t2m_max")
    tmin = expected.get("t2m_min")
    wind = expected.get("ws10m") or 0.0
    rain = details.get("rain_chance_pct") or 0.0
    heavy = details.get("heavy_rain_chance_pct") or 0.0
    heatday = details.get("heat_day_chance_pct") or 0.0
    frost = details.get("frost_chance_pct") or 0.0
    aq = (details.get("air_quality") or {}).get("category") or "—"

    if act == "indoor_gaming":
        score = 98.0
        score -= (heavy * 0.05)
        score -= (rain * 0.02)
        score -= max(0.0, wind - 15.0) * 0.5
    elif act == "indoor":
        score = 94.0
        score -= (heavy * 0.12)
        score -= (rain * 0.05)
        score -= max(0.0, wind - 12.0) * 1.0
        if (tmax or 0) > 37: score -= 3.0
        if (tmin or 99) < 2: score -= 3.0
    elif act == "outdoor_sport":
        score = 85.0
        score -= (rain * 0.65)
        score -= (heavy * 0.9)
        score -= max(0.0, wind - 7.0) * 3.0
        score -= (heatday * 0.35)
        score -= (frost * 0.30)
    elif act == "photo":
        score = 83.0
        score -= (rain * 0.75)
        score -= (heavy * 1.0)
        score -= max(0.0, wind - 5.0) * 3.5
    elif act == "outdoor_gathering":
        score = 80.0
        score -= (rain * 0.7)
        score -= (heavy * 1.0)
        score -= max(0.0, wind - 8.0) * 2.8
        score -= (heatday * 0.25)
    elif act == "work":
        score = 82.0
        score -= (rain * 0.45)
        score -= (heavy * 0.85)
        score -= max(0.0, wind - 9.0) * 2.2
    else:  # generic_outdoor
        score = 82.0
        score -= (rain * 0.55)
        score -= (heavy * 0.9)
        score -= max(0.0, wind - 8.0) * 2.5
        score -= (heatday * 0.3)

    if aq == "Moderate" and not act.startswith("indoor"): score -= 2
    if aq == "Poor" and not act.startswith("indoor"): score -= 8

    score = float(np.clip(score, 0, 100))
    if score >= 85: verdict_key, verdict = "great", "Great"
    elif score >= 70: verdict_key, verdict = "good", "Good"
    elif score >= 55: verdict_key, verdict = "caution", "Caution"
    elif score >= 40: verdict_key, verdict = "risky", "Risky"
    else: verdict_key, verdict = "avoid", "Avoid"

    reasons, prec = _base_reasons_precautions(act, tmax, tmin, wind, rain, heavy, heatday, frost, aq)
    summary = (
        f"{verdict} for "
        f"{('indoor gaming' if act=='indoor_gaming' else ('indoor activity' if act=='indoor' else act.replace('_',' ')))}: "
        f"rain {int(rain)}%, wind {round(wind,1)} m/s, typical max {round((tmax or 0),1)}°C."
    )

    return {
        "verdict": verdict,
        "verdict_key": verdict_key,
        "score": int(round(score)),
        "summary": summary,
        "reasons": reasons[:3],
        "precautions": prec[:3],
        "activity_type": act
    }

# ---------- LLM prompt + JSON handling ----------
JSON_SCHEMA_TEXT = (
    'Return ONLY this JSON:\n'
    '{"verdict":"Great|Good|Caution|Risky|Avoid","verdict_key":"great|good|caution|risky|avoid",'
    '"score":0-100,"summary":"one sentence","reasons":["short","short"],"precautions":["short","short"]}'
)

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

def _repair_jsonish(s: str) -> str:
    s = _strip_code_fences(s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m: s = m.group(0)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    if "'" in s and '"' not in s: s = s.replace("'", '"')
    return s

def _extract_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return json.loads(_repair_jsonish(text))
    except Exception:
        pass
    raise ValueError("Model did not return valid JSON")

def normalize_verdict(v: str):
    v = (v or "").strip().lower()
    if v in ("great","good","caution","risky","avoid"): return v
    mapping = {"excellent":"great","ok":"good","okay":"good","warning":"caution","bad":"avoid","danger":"avoid","poor":"avoid","high risk":"risky","medium":"caution","low":"good"}
    return mapping.get(v, "caution")

def _compact_top_risks(ranked, n=3):
    out = []
    for r in (ranked or [])[:n]:
        out.append({
            "condition": r.get("condition"),
            "label": r.get("label"),
            "probability": r.get("probability"),
            "metric": r.get("metric"),
            "threshold": r.get("threshold"),
            "unit": r.get("unit"),
        })
    return out

def _truncate_text(s, max_chars=400):
    s = (s or "").strip()
    return s[:max_chars]

def _force_summary_alignment(summary: str, base_summary: str, act: str) -> str:
    s = (summary or "").strip()
    if not s:
        return base_summary
    sl = s.lower()
    if act in ("indoor","indoor_gaming"):
        if "outdoor" in sl and "indoor" not in sl:
            return base_summary
    return s

# ---- OpenRouter call (fast, single model, hard timeout, fallback) ----
def call_llm_advice_openrouter(task, date_str, expected, details, ranked):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("openrouter_auth: OPENROUTER_API_KEY not configured on the server.")

    base = compute_activity_assessment(task, expected, details, ranked)
    referer = os.getenv("APP_URL", "http://localhost:3000")
    app_title = os.getenv("APP_NAME", "Weatheryzer")
    model = AI_MODEL_DEFAULT

    ctx = {
        "date": pd.to_datetime(date_str).strftime("%Y-%m-%d"),
        "activity_type": base["activity_type"],
        "expected": expected,
        "chances": {k: details.get(k) for k in [
            "rain_chance_pct","heavy_rain_chance_pct","frost_chance_pct",
            "snow_chance_pct","heat_day_chance_pct","heatwave_chance_pct","cold_spell_chance_pct"]},
        "air_quality": details.get("air_quality"),
        "top_risks": _compact_top_risks(ranked, n=3),
        "fixed": {"score": base["score"], "verdict_key": base["verdict_key"], "verdict": base["verdict"]}
    }

    system = (
        "You format concise, activity-aware weather advice.\n"
        "CRITICAL: Use the fixed score and verdict_key EXACTLY as given (do not change them).\n"
        "Write a single-sentence summary tailored to the user's plan and the activity_type.\n"
        "Return ONLY valid JSON per schema. No markdown."
    )
    user = (
        f"User plan: {_truncate_text(task)}\n\n"
        f"Context JSON:\n{json.dumps(ctx, ensure_ascii=False, separators=(',',':'))}\n\n"
        f"{JSON_SCHEMA_TEXT}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "Referer": referer,
        "X-Title": app_title,
    }
    payload = {
        "model": model,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "temperature": 0.15,
        "max_tokens": 220,
        "top_p": 0.9,
        "response_format": {"type": "json_object"},
        "seed": 42
    }

    start = time.monotonic()
    try:
        remaining = max(1.0, AI_TOTAL_BUDGET - (time.monotonic() - start))
        timeout = min(AI_TIMEOUT, remaining)
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=payload, timeout=timeout)
        if r.status_code in (401,403):
            raise RuntimeError("openrouter_auth: Invalid or missing OpenRouter API key.")
        if r.status_code in (402,429):
            raise RuntimeError("openrouter_quota: OpenRouter quota/credits exceeded for this key.")
        r.raise_for_status()
        jr = r.json()
        text = jr["choices"][0]["message"]["content"]
        data = _extract_json(text)
    except RuntimeError:
        raise
    except Exception as e:
        logging.warning("OpenRouter request failed or timed out (%s); falling back to deterministic advice.", e)
        return {k: base[k] for k in ["verdict","verdict_key","score","summary","reasons","precautions"]}

    data["score"] = base["score"]
    data["verdict_key"] = base["verdict_key"]
    data["verdict"] = base["verdict"]
    data["summary"] = _force_summary_alignment(data.get("summary"), base["summary"], base["activity_type"])
    if not (data.get("reasons") or []): data["reasons"] = base["reasons"]
    if not (data.get("precautions") or []): data["precautions"] = base["precautions"]
    vk = normalize_verdict(data.get("verdict_key") or data.get("verdict"))
    data["verdict_key"] = vk
    data["verdict"] = (data.get("verdict") or vk).title()
    data["score"] = int(float(np.clip(nz(data.get("score"), base["score"]), 0, 100)))
    data["reasons"] = [str(x) for x in (data.get("reasons") or [])][:4]
    data["precautions"] = [str(x) for x in (data.get("precautions") or [])][:4]
    return data

# ---- OpenAI fallback (same contract) ----
def _get_openai_client():
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not installed on server.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY not configured on the server.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return OpenAI(api_key=api_key, timeout=AI_TIMEOUT), model

def call_llm_advice_openai(task, date_str, expected, details, ranked):
    base = compute_activity_assessment(task, expected, details, ranked)
    try:
        client, model = _get_openai_client()
        ctx = {
            "date": pd.to_datetime(date_str).strftime("%Y-%m-%d"),
            "activity_type": base["activity_type"],
            "expected": expected,
            "chances": {k: details.get(k) for k in [
                "rain_chance_pct","heavy_rain_chance_pct","frost_chance_pct",
                "snow_chance_pct","heat_day_chance_pct","heatwave_chance_pct","cold_spell_chance_pct"]},
            "air_quality": details.get("air_quality"),
            "top_risks": _compact_top_risks(ranked, n=3),
            "fixed": {"score": base["score"], "verdict_key": base["verdict_key"], "verdict": base["verdict"]}
        }
        system = ("You format concise weather advice. Use the fixed score and verdict_key EXACTLY as given. "
                  "Return ONLY JSON per schema.")
        user = (
            f"User plan: {_truncate_text(task)}\n\n"
            f"Context JSON:\n{json.dumps(ctx, ensure_ascii=False, separators=(',',':'))}\n\n"
            f"{JSON_SCHEMA_TEXT}"
        )
        resp = client.chat.completions.create(
            model=model, temperature=0.15, max_tokens=220, top_p=0.9,
            response_format={"type": "json_object"}, seed=42,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        text = resp.choices[0].message.content
        data = _extract_json(text)
    except RuntimeError:
        raise
    except Exception as e:
        logging.warning("OpenAI request failed or timed out (%s); falling back to deterministic advice.", e)
        return {k: base[k] for k in ["verdict","verdict_key","score","summary","reasons","precautions"]}

    data["score"] = base["score"]
    data["verdict_key"] = base["verdict_key"]
    data["verdict"] = base["verdict"]
    data["summary"] = _force_summary_alignment(data.get("summary"), base["summary"], base["activity_type"])
    if not (data.get("reasons") or []): data["reasons"] = base["reasons"]
    if not (data.get("precautions") or []): data["precautions"] = base["precautions"]
    vk = normalize_verdict(data.get("verdict_key") or data.get("verdict"))
    data["verdict_key"] = vk
    data["verdict"] = (data.get("verdict") or vk).title()
    data["score"] = int(float(np.clip(nz(data.get("score"), base["score"]), 0, 100)))
    data["reasons"] = [str(x) for x in (data.get("reasons") or [])][:4]
    data["precautions"] = [str(x) for x in (data.get("precautions") or [])][:4]
    return data

def call_llm_advice(task, date_str, expected, details, ranked):
    try:
        if os.getenv("OPENROUTER_API_KEY"):
            return call_llm_advice_openrouter(task, date_str, expected, details, ranked)
        if os.getenv("OPENAI_API_KEY"):
            return call_llm_advice_openai(task, date_str, expected, details, ranked)
    except RuntimeError:
        raise
    except Exception as e:
        logging.warning("AI provider error: %s", e)
    base = compute_activity_assessment(task, expected, details, ranked)
    return {k: base[k] for k in ["verdict","verdict_key","score","summary","reasons","precautions"]}

# -------------- routes --------------
@app.get("/")
def health():
    return "Weatheryzer backend OK"

@app.post("/check-risk")
def check_risk():
    try:
        body = request.get_json(silent=True) or {}
        missing = _require_keys(body, ("lat","lon","date"))
        if missing:
            return jsonify({"error":"bad_request","message":f"Missing keys: {missing}"}), 400

        lat = float(body["lat"]); lon = float(body["lon"])
        date_str = body["date"]
        condition = body.get("condition", "Very Hot")
        window_days = int(body.get("window_days", 7))
        threshold_overrides = body.get("thresholds")

        if parse_any_date(date_str) is None:
            return jsonify({"error": "invalid_date", "message": "Date must be like YYYY-MM-DD"}), 400

        df = fetch_power(lat, lon, start=DEFAULT_START, end=None)
        sub = seasonal_subset(df, date_str, window_days=window_days)

        prob, metric, threshold, unit, series, expected, latest = compute_condition_stats(
            sub, condition, overrides=threshold_overrides, for_date=date_str, full_df=df
        )
        exp_daily, ly_daily, details = daily_snapshot(df, date_str, window_days)

        return jsonify({
            "probability": round(prob*100,1),
            "samples": int(len(series)),
            "metric": metric, "threshold": threshold, "unit": unit,
            "series": series, "expected": expected, "latest_observed": latest,
            "window_days": window_days,
            "lat": round(lat,5), "lon": round(lon,5),
            "daily": {"expected": exp_daily, "last_year": ly_daily, **details},
            "data_coverage": {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            },
        })
    except requests.HTTPError as e:
        return jsonify({"error":"power_api","message":str(e)}), 502
    except Exception as e:
        return jsonify({"error":"server_error","message":str(e)}), 500

@app.post("/best-risk")
def best_risk():
    try:
        body = request.get_json(silent=True) or {}
        missing = _require_keys(body, ("lat","lon","date"))
        if missing:
            return jsonify({"error":"bad_request","message":f"Missing keys: {missing}"}), 400

        lat = float(body["lat"]); lon = float(body["lon"])
        date_str = body["date"]
        window_days = int(body.get("window_days", 7))
        threshold_overrides = body.get("thresholds")

        if parse_any_date(date_str) is None:
            return jsonify({"error": "invalid_date", "message": "Date must be like YYYY-MM-DD"}), 400

        df = fetch_power(lat, lon, start=DEFAULT_START, end=None)
        sub = seasonal_subset(df, date_str, window_days=window_days)
        allres = analyze_all(sub, overrides=threshold_overrides, for_date=date_str, full_df=df)

        exp_daily, ly_daily, details = daily_snapshot(df, date_str, window_days)
        top_with_daily = None
        if allres["top"] is not None:
            top_with_daily = {**allres["top"], "daily": {"expected": exp_daily, "last_year": ly_daily, **details}}

        return jsonify({
            "top": top_with_daily,
            "ranked": allres["ranked"],
            "window_days": window_days,
            "lat": round(lat,5), "lon": round(lon,5),
            "data_coverage": {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            },
        })
    except requests.HTTPError as e:
        return jsonify({"error":"power_api","message":str(e)}), 502
    except Exception as e:
        return jsonify({"error":"server_error","message":str(e)}), 500

@app.post("/ai-advice")
def ai_advice():
    """
    Returns 200 with payload even if AI provider fails (so FE never shows 'Failed to fetch').
    """
    try:
        body = request.get_json(silent=True) or {}
        missing = _require_keys(body, ("lat","lon","date","task"))
        if missing:
            return jsonify({"error":"no_task","message":"Please provide lat, lon, date and a task."}), 200

        lat = float(body["lat"]); lon = float(body["lon"])
        date_str = body["date"]
        task = (body.get("task") or "").strip()
        window_days = int(body.get("window_days", 7))

        if not task:
            return jsonify({"error":"no_task","message":"Please describe your plan/task."}), 200
        if parse_any_date(date_str) is None:
            return jsonify({"error":"invalid_date","message":"Date must be like YYYY-MM-DD"}), 200

        df = fetch_power(lat, lon, start=DEFAULT_START, end=None)
        sub = seasonal_subset(df, date_str, window_days=window_days)
        allres = analyze_all(sub, for_date=date_str, full_df=df)
        expected, last_year, details = daily_snapshot(df, date_str, window_days)

        ai_notice = None
        try:
            advice = call_llm_advice(task, date_str, expected, details, allres.get("ranked", []))
        except RuntimeError as e:
            ai_notice = str(e)
            base = compute_activity_assessment(task, expected, details, allres.get("ranked", []))
            advice = {k: base[k] for k in ["verdict","verdict_key","score","summary","reasons","precautions"]}

        return jsonify({
            "task": task,
            "verdict": advice["verdict"],
            "verdict_key": advice["verdict_key"],
            "emoji": {"great":"🟢","good":"🟩","caution":"🟨","risky":"🟧","avoid":"🟥"}[advice["verdict_key"]],
            "score": advice["score"],
            "summary": advice["summary"],
            "reasons": advice["reasons"],
            "precautions": advice["precautions"],
            "top_risk": (allres.get("ranked") or [None])[0],
            "daily": {"expected": expected, "last_year": last_year, **details},
            "ai_notice": ai_notice,
            "data_coverage": {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            },
        })
    except requests.HTTPError as e:
        return jsonify({"error":"power_api","message":str(e)}), 200
    except Exception as e:
        logging.exception("ai-advice failed")
        return jsonify({"error":"server_error","message":str(e)}), 200

@app.post("/export-csv")
def export_csv():
    body = request.get_json(silent=True) or {}
    missing = _require_keys(body, ("lat","lon","date","condition"))
    if missing:
        return jsonify({"error":"bad_request","message":f"Missing keys: {missing}"}), 400

    lat = float(body["lat"]); lon = float(body["lon"])
    date_str = body["date"]
    condition = body.get("condition","Very Hot")
    window_days = int(body.get("window_days", 7))
    threshold_overrides = body.get("thresholds")

    if parse_any_date(date_str) is None:
        return jsonify({"error":"invalid_date"}), 400

    df = fetch_power(lat, lon, start=DEFAULT_START, end=None)
    sub = seasonal_subset(df, date_str, window_days=window_days)
    p, metric, threshold, unit, series, expected, latest = compute_condition_stats(
        sub, condition, overrides=threshold_overrides, for_date=date_str, full_df=df
    )

    out = pd.DataFrame(series)
    out["metric"] = metric; out["unit"] = unit; out["threshold"] = threshold
    out["expected_median"] = expected.get("median")
    out["expected_p10"] = expected.get("p10")
    out["expected_p90"] = expected.get("p90")
    if latest:
        out["last_year_date"] = latest["date"]; out["last_year_value"] = latest["value"]
    csv = out.to_csv(index=False)
    return csv, 200, {
        "Content-Type": "text/csv",
        "Content-Disposition": f"attachment; filename=weatheryzer_{metric}.csv",
    }

if __name__ == "__main__":
    # Important for Render/Railway/etc.
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
