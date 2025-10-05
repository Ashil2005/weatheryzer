import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Register service worker for PWA (works on HTTPS or localhost)
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    const swUrl = `${process.env.PUBLIC_URL}/sw.js`;
    navigator.serviceWorker
      .register(swUrl)
      .then(reg => {
        // Optional: basic lifecycle logs
        reg.onupdatefound = () => {
          const installing = reg.installing;
          if (installing) {
            installing.onstatechange = () => {
              if (installing.state === 'installed') {
                // If new content is available, a refresh will load it.
                // You could show a toast here if you want.
                // console.log('Service worker installed/updated');
              }
            };
          }
        };
      })
      .catch(() => {
        // Silent fail; PWA still works without SW
      });
  });
}

// CRA perf helper (optional)
reportWebVitals();
