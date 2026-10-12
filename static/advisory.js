/**
 * Renders the advisory dict returned by the backend into a container element.
 * Styled for the dark-mode AQI India dashboard.
 */
function renderAdvisory(el, adv) {
  el.innerHTML = '';
  if (!adv) return;

  const box = document.createElement('div');
  box.style.cssText = [
    'border-left: 4px solid ' + adv.color,
    'background: rgba(255,255,255,.05)',
    'border-radius: 10px',
    'padding: 14px 16px',
    'margin-top: 12px',
    'font-size: .9rem',
  ].join(';');

  // Title row with coloured badge
  const header = document.createElement('div');
  header.style.cssText = 'display:flex;align-items:center;gap:10px;margin-bottom:10px';
  const badge = document.createElement('span');
  badge.style.cssText = 'background:' + adv.color + ';color:#fff;font-weight:700;font-size:.78rem;' +
    'padding:3px 9px;border-radius:20px;white-space:nowrap';
  badge.textContent = adv.bucket + ' · AQI ' + adv.aqi;
  const summary = document.createElement('span');
  summary.style.cssText = 'font-weight:600;color:#e2e8f0';
  summary.textContent = adv.summary;
  header.append(badge, summary);
  box.appendChild(header);

  // Detail rows
  const rows = [
    ['👥 General public', adv.general],
    ['⚕️ Sensitive groups', adv.sensitive],
    ['🏃 Outdoor exercise', adv.outdoor],
    ['😷 Mask', adv.mask],
  ];
  rows.forEach(function ([label, text]) {
    if (!text) return;
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:8px;margin:5px 0;font-size:.86rem;color:#cbd5e1';
    const b = document.createElement('span');
    b.style.cssText = 'color:#94a3b8;min-width:140px;flex-shrink:0';
    b.textContent = label + ':';
    const v = document.createElement('span');
    v.textContent = text;
    row.append(b, v);
    box.appendChild(row);
  });

  // Cigarette equivalent
  if (adv.cigarettes_per_day !== null && adv.cigarettes_per_day !== undefined) {
    const c = document.createElement('div');
    c.style.cssText = 'margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.07);' +
      'font-size:.82rem;color:#94a3b8';
    c.innerHTML = '🚬 PM2.5 ≈ <strong style="color:#fcd34d">' + adv.cigarettes_per_day +
      ' cigarette(s)</strong> per day. ' + (adv.cigarette_note || '');
    box.appendChild(c);
  }

  el.appendChild(box);
}
