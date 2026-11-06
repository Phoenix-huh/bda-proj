/**
 * trends_charts.js  –  shared Chart.js render functions for the AQI trends charts.
 *
 * Renders Charts A, B, C, E parameterised by canvas element ID so the same
 * functions can be used on both /trends and /covid pages.
 *
 * Requires Chart.js 4.4.3 to be loaded before this script.
 * Chart D (named-cities line) is NOT here – it has page-specific dropdown
 * state and lives only in trends.html.
 */

// ── Error Bars Plugin ─────────────────────────────────────────────────────────
const errorBarsPlugin = {
  id: 'errorBarsPlugin',
  afterDatasetsDraw(chart, args, options) {
    if (!options || !options.errorBars) return;
    const { ctx, scales: { x, y } } = chart;
    const meta = chart.getDatasetMeta(0);
    if (!meta || !meta.data) return;

    const errorBars = options.errorBars;
    ctx.save();
    ctx.strokeStyle = options.color || '#fca5a5';
    ctx.lineWidth = options.lineWidth || 2;

    meta.data.forEach((bar, index) => {
      const eb = errorBars[index];
      if (!eb || eb.low === undefined || eb.high === undefined) return;

      const xPos = bar.x;
      const yLow  = y.getPixelForValue(eb.low);
      const yHigh = y.getPixelForValue(eb.high);
      const capWidth = options.capWidth || 8;

      // Vertical stem
      ctx.beginPath();
      ctx.moveTo(xPos, yLow);
      ctx.lineTo(xPos, yHigh);
      ctx.stroke();

      // High cap
      ctx.beginPath();
      ctx.moveTo(xPos - capWidth / 2, yHigh);
      ctx.lineTo(xPos + capWidth / 2, yHigh);
      ctx.stroke();

      // Low cap
      ctx.beginPath();
      ctx.moveTo(xPos - capWidth / 2, yLow);
      ctx.lineTo(xPos + capWidth / 2, yLow);
      ctx.stroke();
    });
    ctx.restore();
  }
};

// Guard against double-registration when this script is loaded multiple times.
if (!Chart.registry.plugins.get('errorBarsPlugin')) {
  Chart.register(errorBarsPlugin);
}

// ── Shared style constants ────────────────────────────────────────────────────
const chartFont   = { family: "'Inter', system-ui, sans-serif", size: 12 };
const gridColor   = 'rgba(255, 255, 255, 0.08)';
const textColor   = '#94a3b8';

// Registry to track Chart instances so we can destroy & re-render safely.
const _chartInstances = {};

function _newChart(canvasId, config) {
  if (_chartInstances[canvasId]) {
    _chartInstances[canvasId].destroy();
  }
  const ctx = document.getElementById(canvasId);
  if (!ctx) {
    console.warn('trends_charts.js: canvas not found:', canvasId);
    return null;
  }
  _chartInstances[canvasId] = new Chart(ctx.getContext('2d'), config);
  return _chartInstances[canvasId];
}

// ── Chart A: Median Window AQI Per Year (2020 highlighted) ───────────────────
/**
 * @param {Object} data  – /api/aqi-trends JSON
 * @param {string} canvasId  – id of the <canvas> element to render into
 */
function renderChartA(data, canvasId) {
  canvasId = canvasId || 'chart-a-median-aqi';
  const years  = data.years.map(String);
  const values = years.map(y => data.median_window_aqi[y]);

  const bgColors     = years.map(y => y === '2020' ? 'rgba(56, 189, 248, 0.95)' : 'rgba(59, 130, 246, 0.55)');
  const borderColors = years.map(y => y === '2020' ? '#38bdf8' : '#3b82f6');

  return _newChart(canvasId, {
    type: 'bar',
    data: {
      labels: years,
      datasets: [{
        label: 'Median Window AQI',
        data: values,
        backgroundColor: bgColors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Median AQI: ${ctx.parsed.y}${ctx.label === '2020' ? ' (Lockdown Window)' : ''}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: chartFont },
          title: { display: true, text: 'Year', color: textColor, font: chartFont }
        },
        y: {
          beginAtZero: true,
          suggestedMax: 160,
          grid: { color: gridColor },
          ticks: { color: textColor, font: chartFont },
          title: { display: true, text: 'AQI Value (Index)', color: textColor, font: chartFont }
        }
      }
    }
  });
}

// ── Chart B: Percent Change vs 2020 with Error Bars ──────────────────────────
/**
 * @param {Object} data
 * @param {string} canvasId
 */
function renderChartB(data, canvasId) {
  canvasId = canvasId || 'chart-b-pct-change';
  const years  = ['2021', '2022', '2023', '2024', '2025'];
  const values = years.map(y => data.pct_change_vs_2020[y].median_pct_change);
  const errorBars = years.map(y => ({
    low:  data.pct_change_vs_2020[y].ci_low,
    high: data.pct_change_vs_2020[y].ci_high
  }));

  return _newChart(canvasId, {
    type: 'bar',
    data: {
      labels: years,
      datasets: [{
        label: 'Median % Change vs 2020',
        data: values,
        backgroundColor: 'rgba(239, 68, 68, 0.65)',
        borderColor: '#ef4444',
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const y    = ctx.label;
              const item = data.pct_change_vs_2020[y];
              return [
                `Median Change: +${item.median_pct_change}%`,
                `95% CI: [${item.ci_low}%, ${item.ci_high}%]`,
                `Wilcoxon p: ${item.wilcoxon_p.toExponential(2)}`
              ];
            }
          }
        },
        errorBarsPlugin: {
          errorBars: errorBars,
          color: '#fca5a5',
          lineWidth: 2,
          capWidth: 10
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: chartFont },
          title: { display: true, text: 'Year', color: textColor, font: chartFont }
        },
        y: {
          beginAtZero: true,
          suggestedMax: 85,
          grid: { color: gridColor },
          ticks: {
            color: textColor,
            font: chartFont,
            callback: (v) => `+${v}%`
          },
          title: { display: true, text: 'Percent Change vs 2020 (%)', color: textColor, font: chartFont }
        }
      }
    }
  });
}

// ── Chart C: Category Shares per Year (Stacked Bar) ──────────────────────────
/**
 * @param {Object} data
 * @param {string} canvasId
 */
function renderChartC(data, canvasId) {
  canvasId = canvasId || 'chart-c-category-shares';
  const years = data.years.map(String);

  const goodSat = years.map(y => data.category_shares[y].good_satisfactory_pct);
  const moderate = years.map(y => data.category_shares[y].moderate_pct);
  const poor     = years.map(y => data.category_shares[y].poor_pct);
  const vpSevere = years.map(y => data.category_shares[y].very_poor_severe_pct);

  return _newChart(canvasId, {
    type: 'bar',
    data: {
      labels: years,
      datasets: [
        { label: 'Good + Satisfactory (≤100)', data: goodSat, backgroundColor: '#10b981', borderRadius: 3 },
        { label: 'Moderate (101–200)',          data: moderate, backgroundColor: '#f59e0b', borderRadius: 3 },
        { label: 'Poor (201–300)',              data: poor,     backgroundColor: '#f97316', borderRadius: 3 },
        { label: 'Very Poor + Severe (>300)',   data: vpSevere, backgroundColor: '#ef4444', borderRadius: 3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { color: '#e2e8f0', font: { family: "'Inter', sans-serif", size: 11 }, boxWidth: 12 }
        },
        tooltip: {
          callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%` }
        }
      },
      scales: {
        x: {
          stacked: true,
          grid: { display: false },
          ticks: { color: textColor, font: chartFont },
          title: { display: true, text: 'Year', color: textColor, font: chartFont }
        },
        y: {
          stacked: true,
          beginAtZero: true,
          max: 100,
          grid: { color: gridColor },
          ticks: { color: textColor, font: chartFont, callback: (v) => `${v}%` },
          title: { display: true, text: 'Share of Panel City-Days (%)', color: textColor, font: chartFont }
        }
      }
    }
  });
}

// ── Chart E: Pre-Window vs Window Median AQI ─────────────────────────────────
/**
 * @param {Object} data
 * @param {string} canvasId
 */
function renderChartE(data, canvasId) {
  canvasId = canvasId || 'chart-e-pre-vs-window';
  const years   = data.years.map(String);
  const preVals = years.map(y => data.pre_window_median_aqi[y]);
  const winVals = years.map(y => data.median_window_aqi[y]);

  return _newChart(canvasId, {
    type: 'bar',
    data: {
      labels: years,
      datasets: [
        {
          label: 'Pre-Window (1 Jan – 24 Mar)',
          data: preVals,
          backgroundColor: 'rgba(148, 163, 184, 0.65)',
          borderColor: '#94a3b8',
          borderWidth: 1.5,
          borderRadius: 4
        },
        {
          label: 'Window (25 Mar – 31 May)',
          data: winVals,
          backgroundColor: 'rgba(56, 189, 248, 0.75)',
          borderColor: '#38bdf8',
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { color: '#e2e8f0', font: chartFont, boxWidth: 12 }
        },
        tooltip: {
          callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y} AQI` }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: textColor, font: chartFont },
          title: { display: true, text: 'Year', color: textColor, font: chartFont }
        },
        y: {
          beginAtZero: true,
          suggestedMax: 180,
          grid: { color: gridColor },
          ticks: { color: textColor, font: chartFont },
          title: { display: true, text: 'Median AQI (Index)', color: textColor, font: chartFont }
        }
      }
    }
  });
}

