const metrics = [
  {
    tag: 'baseline',
    title: 'Notebook baseline',
    value: 'QWK 0.6812',
    detail: '529 / 540 valid, Accuracy 0.6427 on old test_ua.',
  },
  {
    tag: 'binary',
    title: 'Binary baseline',
    value: 'Acc 0.7608',
    detail: '5000 / 5000 valid, QWK 0.5066 on scientsbank.jsonl.',
  },
  {
    tag: 'attack',
    title: 'Banked heldout100',
    value: 'ASR 0.75',
    detail: '85 / 100 incorrect samples flipped to correct.',
  },
  {
    tag: 'ablation',
    title: 'No-bank heldout100',
    value: 'ASR 0.02',
    detail: '12 / 100 flipped to correct, most moved to contradictory.',
  },
];

const asrData = [
  { name: 'banked', value: 75, note: 'pure GCG + transfer bank' },
  { name: 'contrabank', value: 76, note: 'pure GCG + contradictory bank' },
  { name: 'hybrid', value: 90, note: 'upper bound, not pure GCG' },
  { name: 'no-bank', value: 2, note: 'same setup without suffix bank' },
];

const compareRuns = {
  banked: {
    label: 'banked',
    badge: 'transfer suffix bank enabled',
    matrix: [
      ['True / Pred', 'correct', 'contradictory', 'incorrect'],
      ['correct', 0, 0, 0],
      ['contradictory', 0, 0, 0],
      ['incorrect', 85, 12, 3],
    ],
    dist: [
      { name: 'correct', value: 85, color: 'c-green' },
      { name: 'contradictory', value: 12, color: 'c-amber' },
      { name: 'incorrect', value: 3, color: 'c-red' },
    ],
  },
  nobank: {
    label: 'no-bank',
    badge: 'transfer suffix bank removed',
    matrix: [
      ['True / Pred', 'correct', 'contradictory', 'incorrect'],
      ['correct', 0, 0, 0],
      ['contradictory', 0, 0, 0],
      ['incorrect', 12, 46, 42],
    ],
    dist: [
      { name: 'correct', value: 12, color: 'c-green' },
      { name: 'contradictory', value: 46, color: 'c-amber' },
      { name: 'incorrect', value: 42, color: 'c-red' },
    ],
  },
};

function barRow(label, value, color, total = 100, suffix = '%') {
  const pct = Math.max(0, Math.min(100, (value / total) * 100));
  return `
    <div class="bar-row">
      <div class="bar-label">
        <strong>${label}</strong>
        <span>${value}${suffix}</span>
      </div>
      <div class="bar-track" aria-hidden="true">
        <div class="bar-seg ${color}" style="width:${pct}%"></div>
      </div>
    </div>
  `;
}

function renderMetrics() {
  const grid = document.getElementById('metric-grid');
  grid.innerHTML = metrics.map((m) => `
    <article class="metric-card">
      <span class="metric-tag">${m.tag}</span>
      <h4>${m.title}</h4>
      <div class="metric-value">${m.value}</div>
      <div class="metric-detail">${m.detail}</div>
    </article>
  `).join('');
}

function renderASRBars() {
  const target = document.getElementById('asr-bars');
  target.innerHTML = asrData.map((r) => barRow(`${r.name}`, r.value, r.name === 'hybrid' ? 'c-slate' : (r.name === 'banked' ? 'c-green' : 'c-red'))).join('') +
    `
      <div class="bar-legend">
        ${asrData.map((r) => `<span><i class="bar-dot ${r.name === 'hybrid' ? 'c-slate' : (r.name === 'banked' ? 'c-green' : 'c-red')}"></i>${r.note}</span>`).join('')}
      </div>
    `;
}

function renderMatrix(view) {
  const data = compareRuns[view];
  document.getElementById('matrix-title').textContent = `Attack confusion matrix: ${data.label}`;
  document.getElementById('matrix-badge').textContent = data.badge;
  document.getElementById('matrix-legend').textContent = view === 'banked' ? '85 / 12 / 3' : '12 / 46 / 42';

  const table = document.getElementById('matrix-table');
  const rows = data.matrix.map((row, idx) => {
    if (idx === 0) {
      return `
        <div class="matrix-grid">
          ${row.map((cell) => `<div class="matrix-cell matrix-headcell">${cell}</div>`).join('')}
        </div>
      `;
    }
    if (idx === 1 || idx === 2) {
      return `
        <div class="matrix-grid">
          <div class="matrix-cell matrix-rowhead">${row[0]}</div>
          <div class="matrix-cell">0</div>
          <div class="matrix-cell">0</div>
          <div class="matrix-cell">0</div>
        </div>
      `;
    }
    return `
      <div class="matrix-grid">
        <div class="matrix-cell matrix-rowhead">${row[0]}</div>
        <div class="matrix-cell"><span class="matrix-score">${row[1]}</span></div>
        <div class="matrix-cell"><span class="matrix-score">${row[2]}</span></div>
        <div class="matrix-cell"><span class="matrix-score">${row[3]}</span></div>
      </div>
    `;
  }).join('');
  table.innerHTML = rows;

  const bar = document.getElementById('matrix-bar');
  bar.innerHTML = data.dist.map((d) => barRow(d.name, d.value, d.color)).join('') +
    `
      <div class="bar-legend">
        ${data.dist.map((d) => `<span><i class="bar-dot ${d.color}"></i>${d.name}</span>`).join('')}
      </div>
    `;
}

function bindToggle() {
  document.querySelectorAll('.toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      renderMatrix(btn.dataset.view);
    });
  });
}

renderMetrics();
renderASRBars();
renderMatrix('banked');
bindToggle();
