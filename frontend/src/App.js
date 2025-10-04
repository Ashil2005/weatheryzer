import React, { useMemo, useState } from "react";
import { Line, Bar, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement,
  Legend, Tooltip,
} from "chart.js";
import DatePicker from "react-datepicker";
import { parse, isValid, format } from "date-fns";
import "react-datepicker/dist/react-datepicker.css";
import "./App.css";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Legend, Tooltip);

// ---- helpers
function parseLatLonSmart(str) {
  if (!str) return [NaN, NaN];
  const m = str
    .toUpperCase()
    .replace(/AND/g, ",")
    .match(/(-?\d+(?:\.\d+)?)\s*°?\s*([NS])?[^0-9\-+]*(-?\d+(?:\.\d+)?)\s*°?\s*([EW])?/);
  if (m) {
    let lat = parseFloat(m[1]), lon = parseFloat(m[3]);
    const ns = m[2], ew = m[4];
    if (ns === "S") lat = -Math.abs(lat);
    if (ns === "N") lat = Math.abs(lat);
    if (ew === "W") lon = -Math.abs(lon);
    if (ew === "E") lon = Math.abs(lon);
    return [lat, lon];
  }
  const parts = str.split(",").map(s => s.trim());
  if (parts.length === 2) return [parseFloat(parts[0]), parseFloat(parts[1])];
  return [NaN, NaN];
}
function parseToDate(v) {
  if (!v || typeof v !== "string") return null;
  const s = v.trim(); if (!s) return null;
  const fmts = ["yyyy-MM-dd", "dd-MM-yyyy", "dd/MM/yyyy", "MM/dd/yyyy"];
  for (const f of fmts) {
    const d = parse(s, f, new Date());
    if (isValid(d)) return d;
  }
  return null;
}
const fmt = (x, d=1) => (x === null || x === undefined || Number.isNaN(+x) ? "—" : (+x).toFixed(d));
const API_BASE = "http://127.0.0.1:5000";

// ---- gauge
function Gauge({ value }) {
  const data = {
    labels: ["Risk", "Other"],
    datasets: [{
      data: [value, 100 - value],
      backgroundColor: [
        value >= 50 ? "rgba(239,68,68,0.95)" : "rgba(37,99,235,0.95)",
        "rgba(148,163,184,0.15)"
      ],
      borderWidth: 0
    }]
  };
  return (
    <div className="ring">
      <Doughnut data={data} options={{ cutout: "70%", plugins: { legend: { display: false } } }} />
      <div className="ringText">{value}%</div>
    </div>
  );
}

export default function App() {
  const [place, setPlace] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [location, setLocation] = useState("");
  const [date, setDate] = useState("");
  const [dateObj, setDateObj] = useState(null);
  const [condition, setCondition] = useState("Very Hot");

  const [result, setResult] = useState(null);
  const [ranked, setRanked] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  // NEW: AI advice state
  const [task, setTask] = useState("");
  const [advice, setAdvice] = useState(null);

  // derived: is the date valid?
  const validDate = useMemo(() => !!parseToDate(date), [date]);

  // search
  const searchPlace = async (q) => {
    setPlace(q);
    setErr("");
    if (q.length < 3) { setSuggestions([]); return; }
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=5&addressdetails=1`;
      const res = await fetch(url, { headers: { "Accept-Language": "en" } });
      const data = await res.json();
      setSuggestions(data.map(d => ({ name: d.display_name, lat: parseFloat(d.lat), lon: parseFloat(d.lon) })));
    } catch { setErr("Could not search places. Try again."); }
  };
  const chooseSuggestion = (s) => {
    setPlace(s.name);
    setLocation(`${s.lat.toFixed(5)},${s.lon.toFixed(5)}`);
    setSuggestions([]);
  };
  const useMyLocation = () => {
    setErr("");
    if (!navigator.geolocation) { setErr("Geolocation not supported."); return; }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        setLocation(`${latitude.toFixed(5)},${longitude.toFixed(5)}`);
      },
      () => setErr("Could not get your location.")
    );
  };

  // date handlers
  const onDateChange = (d) => {
    setDateObj(d);
    setResult(null); setRanked(null); setAdvice(null); setErr("");
    setDate(d ? format(d,"yyyy-MM-dd") : "");
  };
  const onDateRaw = (e) => {
    const raw = typeof e === "string" ? e : (e?.target?.value ?? "");
    const v = raw.replace(/[^\d/-]/g, "");
    const d = parseToDate(v);
    if (d) { setDateObj(d); setDate(format(d,"yyyy-MM-dd")); }
    else   { setDate(v); }
    setResult(null); setRanked(null); setAdvice(null); setErr("");
  };

  // validation
  const requireLatLonDate = () => {
    let [lat, lon] = parseLatLonSmart(location);
    if (Number.isNaN(lat) || Number.isNaN(lon)) { setErr("Enter/select a valid location."); return null; }
    if (!validDate) { setErr("Pick a valid date (YYYY-MM-DD)."); return null; }
    return { lat, lon };
  };

  const handleJson = async (res) => {
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); if (j?.message) msg = j.message; } catch {}
      throw new Error(msg);
    }
    return res.json();
  };

  // API
  const handleSubmit = async (e) => {
    e.preventDefault();
    setErr(""); setResult(null); setRanked(null); setAdvice(null);
    const v = requireLatLonDate(); if (!v) return;
    const { lat, lon } = v;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/check-risk`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon, date, condition }),
      });
      const data = await handleJson(res);
      setResult({ ...data, lat, lon, date, condition });
    } catch (e) { setErr(`Request failed: ${e.message}`); }
    finally { setLoading(false); }
  };

  const analyzeAll = async () => {
    setErr(""); setResult(null); setRanked(null); setAdvice(null);
    const v = requireLatLonDate(); if (!v) return;
    const { lat, lon } = v;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/best-risk`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon, date }),
      });
      const data = await handleJson(res);
      setRanked(data.ranked || []);
      if (data.top) {
        setCondition(data.top.condition);
        setResult({ ...data.top, lat, lon, date, condition: data.top.condition, window_days: data.window_days });
      } else {
        setErr("No historical samples for that date window.");
      }
    } catch (e) { setErr(`Request failed: ${e.message}`); }
    finally { setLoading(false); }
  };

  const viewCondition = async (cond) => {
    setCondition(cond);
    const v = requireLatLonDate(); if (!v) return;
    const { lat, lon } = v;
    setLoading(true); setErr("");
    try {
      const res = await fetch(`${API_BASE}/check-risk`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon, date, condition: cond }),
      });
      const data = await handleJson(res);
      setResult({ ...data, lat, lon, date, condition: cond });
    } catch (e) { setErr(`Could not load series for ${cond}: ${e.message}`); }
    finally { setLoading(false); }
  };

  const downloadCSV = async () => {
    if (!result) return;
    const { lat, lon, date, condition } = result;
    const res = await fetch(`${API_BASE}/export-csv`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, date, condition }),
    });
    if (!res.ok) { setErr("CSV download failed."); return; }
    const text = await res.text();
    const blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `weatheryzer_${result.metric}.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  // NEW: AI advice
  const getAdvice = async () => {
    setErr(""); setAdvice(null);
    const v = requireLatLonDate(); if (!v) return;
    if (!task.trim()) { setErr("Describe your plan in the text box."); return; }
    const { lat, lon } = v;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/ai-advice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon, date, task }),
      });
      const data = await handleJson(res);
      setAdvice(data);
      if (!result?.daily && data?.daily) {
        setResult(prev => prev ? { ...prev, daily: data.daily } : prev);
      }
    } catch (e) { setErr(`Advice failed: ${e.message}`); }
    finally { setLoading(false); }
  };

  // charts
  const lineData = result?.series?.length ? {
    labels: result.series.map(d => d.date),
    datasets: [
      {
        label: `${result.metric} (${result.unit})`,
        data: result.series.map(d => d.value),
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56,189,248,0.15)",
        tension: 0.25, pointRadius: 0,
      },
      {
        label: `Threshold (${result.threshold} ${result.unit})`,
        data: result.series.map(() => result.threshold),
        borderColor: "#f87171",
        borderDash: [6, 6],
        pointRadius: 0,
      },
    ],
  } : null;

  const bars = ranked ? {
    labels: ranked.map(r => r.label),
    datasets: [{
      label: "Probability (%)",
      data: ranked.map(r => r.probability),
      backgroundColor: ranked.map(r => r.probability >= 50 ? "rgba(248,113,113,0.7)" : "rgba(56,189,248,0.7)")
    }]
  } : null;

  const barOptions = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: { y: { min: 0, max: 100 } },
    onClick: (_evt, elements) => {
      if (!elements?.length) return;
      const idx = elements[0].index;
      const item = ranked?.[idx];
      if (item) viewCondition(item.condition);
    }
  };

  const daily = result?.daily;

  return (
    <div className="shell">
      <header className="hero">
        <h1>🌍 Weatheryzer</h1>
        <p>Historical likelihood of <b>Very Hot</b>, <b>Very Cold</b>, <b>Very Wet</b>, <b>Very Windy</b>, or <b>Very Uncomfortable</b> for any location & date.</p>
      </header>

      <div className="card">
        <div className="row">
          <div className="grow">
            <label>🔎 Place</label>
            <input
              value={place}
              onChange={(e) => { setRanked(null); setResult(null); setAdvice(null); searchPlace(e.target.value); }}
              placeholder="e.g. New York City, Nairobi National Park, Bangalore"
            />
            {suggestions.length > 0 && (
              <div className="suggest">
                {suggestions.map((s, i) => (
                  <div key={i} onClick={() => chooseSuggestion(s)}>
                    {s.name} <span className="muted">({s.lat.toFixed(3)}, {s.lon.toFixed(3)})</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <button className="ghost useBtn" onClick={useMyLocation}>📍 Use my location</button>
        </div>

        <form onSubmit={handleSubmit} className="grid2">
          <div>
            <label>📍 Location (lat,lon)</label>
            <input value={location} onChange={e => { setLocation(e.target.value); setRanked(null); setResult(null); setAdvice(null); }} placeholder="auto-filled or type lat,lon" />
          </div>
          <div>
            <label>📅 Date (any day of year — not a forecast)</label>
            <DatePicker
              selected={dateObj}
              onChange={onDateChange}
              onChangeRaw={onDateRaw}
              dateFormat="yyyy-MM-dd"
              placeholderText="YYYY-MM-DD"
              className="dpInput"
              isClearable
              showPopperArrow
            />
            {!validDate && date && (
              <div className="small" style={{ color: "#fca5a5", marginTop: 4 }}>
                Enter a valid date (YYYY-MM-DD).
              </div>
            )}
          </div>
          <div>
            <label>☁️ Condition</label>
            <select value={condition} onChange={e => viewCondition(e.target.value)}>
              <option>Very Hot</option>
              <option>Very Cold</option>
              <option>Very Windy</option>
              <option>Very Wet</option>
              <option>Very Uncomfortable</option>
            </select>
          </div>
          <div className="row">
            <button type="submit" disabled={loading || !validDate}>{loading ? "Checking..." : "🔍 Check Weather Risk"}</button>
            <button type="button" className="ghost" onClick={analyzeAll} disabled={loading || !validDate}>{loading ? "Analyzing..." : "🧠 Analyze All Conditions"}</button>
          </div>
        </form>

        {/* NEW: AI task input */}
        <div className="panel" style={{ marginTop: 12 }}>
          <label>🤖 What are you planning to do on that day?</label>
          <textarea
            rows={3}
            placeholder="e.g., 20 km bike ride at noon; outdoor wedding photoshoot; school sports day; market stall..."
            value={task}
            onChange={(e) => setTask(e.target.value)}
          />
          <div className="row" style={{ marginTop: 8 }}>
            <button type="button" onClick={getAdvice} disabled={loading || !validDate || !task.trim()}>
              {loading ? "Thinking..." : "🤖 Get AI Advice"}
            </button>
            <span className="small muted">AI uses the historical stats for that date/location.</span>
          </div>
        </div>

        {err && <p className="error">{err}</p>}
      </div>

      {ranked && (
        <div className="card">
          <h3>Most likely condition: <span className="accent">{ranked[0]?.label}</span> <span className="accent2">({ranked[0]?.probability}%)</span></h3>
          {bars && <div className="panel"><Bar data={bars} options={barOptions} /></div>}
          <div className="chips">
            {ranked.map((r, i) => (
              <button key={i} onClick={() => viewCondition(r.condition)} className="chip">
                {r.label}: {r.probability}%
              </button>
            ))}
          </div>
        </div>
      )}

      {result && (
        <div className="card">
          <div className="grid2">
            <div>
              <h3>Probability of <b>{result.condition}</b> on <b>{result.date}</b></h3>
              <Gauge value={result.probability} />
              <p className="muted small">
                Based on {result.samples} historical samples (±{result.window_days}-day seasonal window).
              </p>
            </div>

            <div className="panel">
              <h4>Expected vs Last Year</h4>
              <p>
                Expected <b>{result.metric}</b> (<span className="muted">{result.unit}</span>) on this day:
                {" "}<b>{fmt(result.expected?.median, 2)}</b>
                {" "}<span className="muted">[p10–p90: {fmt(result.expected?.p10,2)}–{fmt(result.expected?.p90,2)}]</span>
              </p>
              <p>
                Last year on this day:
                {" "}<b>{fmt(result.latest_observed?.value,2)} {result.unit}</b>
                {result.latest_observed?.date && <span className="muted"> ({result.latest_observed.date})</span>}
              </p>
              <button onClick={downloadCSV}>⬇️ Download CSV (subset)</button>
            </div>
          </div>

          {/* Daily snapshot+ */}
          {daily && (
            <div className="panel">
              <h4>🌤️ Daily snapshot & event chances</h4>
              <div className="grid2">
                <div>
                  <p>🌡️ <b>Temperature</b> (°C)</p>
                  <ul className="small muted">
                    <li>Mean: <b>{fmt(daily.expected?.t2m)}</b></li>
                    <li>High: <b>{fmt(daily.expected?.t2m_max)}</b></li>
                    <li>Low: <b>{fmt(daily.expected?.t2m_min)}</b></li>
                  </ul>
                </div>
                <div>
                  <p>🌧️ <b>Precipitation</b></p>
                  <ul className="small muted">
                    <li>Amount (median): <b>{fmt(daily.expected?.precip)} mm</b></li>
                    <li>Chance ≥1 mm: <b>{fmt(daily.rain_chance_pct,0)}%</b></li>
                    <li>Chance heavy rain: <b>{fmt(daily.heavy_rain_chance_pct,0)}%</b></li>
                  </ul>
                </div>
                <div>
                  <p>❄️ <b>Cold/Snow</b></p>
                  <ul className="small muted">
                    <li>Frost chance: <b>{fmt(daily.frost_chance_pct,0)}%</b></li>
                    <li>Snow chance: <b>{fmt(daily.snow_chance_pct,0)}%</b></li>
                    <li>Cold spell (3-day) chance: <b>{fmt(daily.cold_spell_chance_pct,0)}%</b></li>
                  </ul>
                </div>
                <div>
                  <p>🔥 <b>Heat</b></p>
                  <ul className="small muted">
                    <li>Hot-day chance (≥35 °C): <b>{fmt(daily.heat_day_chance_pct,0)}%</b></li>
                    <li>Heatwave (3-day) chance: <b>{fmt(daily.heatwave_chance_pct,0)}%</b></li>
                  </ul>
                </div>
                <div>
                  <p>☁️ <b>Cloud / 💨 Wind / 💧 Humidity</b></p>
                  <ul className="small muted">
                    <li>Cloudiness (median): <b>{fmt(daily.expected?.cloud_pct,0)}%</b></li>
                    <li>Wind (median): <b>{fmt(daily.expected?.ws10m)} m/s</b></li>
                    <li>Humidity (median): <b>{fmt(daily.expected?.rh2m,0)}%</b></li>
                  </ul>
                </div>
                <div>
                  <p>🌫️ <b>Air quality / Dust proxy</b></p>
                  <ul className="small muted">
                    <li>AOD 550 nm (median): <b>{fmt(daily.air_quality?.aod550_median,3)}</b></li>
                    <li>Category: <b>{daily.air_quality?.category ?? "—"}</b></li>
                  </ul>
                </div>
              </div>

              {daily.last_year && (
                <p className="small muted" style={{ marginTop: 8 }}>
                  Last year ({daily.last_year.date}): mean {fmt(daily.last_year.t2m)}°C,
                  high {fmt(daily.last_year.t2m_max)}°C, low {fmt(daily.last_year.t2m_min)}°C,
                  rain {fmt(daily.last_year.precip)} mm, clouds {fmt(daily.last_year.cloud_pct,0)}%,
                  wind {fmt(daily.last_year.ws10m)} m/s, humidity {fmt(daily.last_year.rh2m,0)}%,
                  AOD {fmt(daily.last_year.aod550,3)}.
                </p>
              )}
              <p className="small muted" style={{ marginTop: 6 }}>
                Air-quality/dust uses satellite AOD as a coarse proxy (not a regulatory AQI).
              </p>
            </div>
          )}
        </div>
      )}

      {/* NEW: AI Advice card */}
      {advice && (
        <div className="card">
          <div className={`adviceBadge ${advice.verdict_key}`}>
            <span style={{ fontSize: 18 }}>{advice.emoji}</span>
            {advice.verdict}
          </div>
          <div className="grid2" style={{ marginTop: 12 }}>
            <div className="panel">
              <h4>Your plan</h4>
              <p className="muted">{advice.task}</p>
              <h4 style={{ marginTop: 10 }}>Summary</h4>
              <p>{advice.summary}</p>
            </div>
            <div className="panel" style={{ textAlign: "center" }}>
              <h4>Suitability score</h4>
              <Gauge value={advice.score} />
              <p className="small muted">0 = avoid, 100 = great</p>
            </div>
          </div>
          <div className="grid2" style={{ marginTop: 12 }}>
            <div className="panel">
              <h4>Why</h4>
              <ul className="adviceList">{advice.reasons?.map((r,i)=><li key={i}>{r}</li>)}</ul>
            </div>
            <div className="panel">
              <h4>Precautions</h4>
              <ul className="adviceList">{advice.precautions?.map((r,i)=><li key={i}>{r}</li>)}</ul>
            </div>
          </div>
        </div>
      )}

      {/* Bottom "Day-at-a-glance" tiles */}
      {daily && (
        <div className="card">
          <h3>📊 Day-at-a-glance</h3>
          <div className="facts">
            <div className="fact">
              <h5>Temperature</h5>
              <div className="kpi">{fmt(daily.expected?.t2m)}°C</div>
              <div className="muted small">High {fmt(daily.expected?.t2m_max)} / Low {fmt(daily.expected?.t2m_min)}°C</div>
            </div>
            <div className="fact">
              <h5>Precipitation</h5>
              <div className="kpi">{fmt(daily.expected?.precip)} mm</div>
              <div className="muted small">≥1mm: {fmt(daily.rain_chance_pct,0)}% • Heavy: {fmt(daily.heavy_rain_chance_pct,0)}%</div>
            </div>
            <div className="fact">
              <h5>Cloud cover</h5>
              <div className="kpi">{fmt(daily.expected?.cloud_pct,0)}%</div>
            </div>
            <div className="fact">
              <h5>Wind</h5>
              <div className="kpi">{fmt(daily.expected?.ws10m)} m/s</div>
              <div className="muted small">Humidity {fmt(daily.expected?.rh2m,0)}%</div>
            </div>
            <div className="fact">
              <h5>Heat</h5>
              <div className="kpi">{fmt(daily.heat_day_chance_pct,0)}%</div>
              <div className="muted small">Heatwave (3-day): {fmt(daily.heatwave_chance_pct,0)}%</div>
            </div>
            <div className="fact">
              <h5>Cold/Snow</h5>
              <div className="kpi">{fmt(daily.frost_chance_pct,0)}%</div>
              <div className="muted small">Snow: {fmt(daily.snow_chance_pct,0)}% • Cold spell: {fmt(daily.cold_spell_chance_pct,0)}%</div>
            </div>
            <div className="fact">
              <h5>Air quality (AOD)</h5>
              <div className="kpi">{fmt(daily.air_quality?.aod550_median,3)}</div>
              <div className="muted small">{daily.air_quality?.category ?? "—"}</div>
            </div>
          </div>
        </div>
      )}

      <footer className="muted small">
        Data: NASA POWER (daily). These are historical statistics (not a forecast).
      </footer>
    </div>
  );
}
