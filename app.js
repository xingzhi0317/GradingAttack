const evidenceMetrics = [
  {
    tag: 'baseline',
    title: 'Three-class baseline',
    value: 'QWK 0.6812',
    detail: 'Old SciEntsBank test_ua: 529 / 540 valid, Accuracy 0.6427.',
  },
  {
    tag: 'validated attack',
    title: 'Strong GCG flip',
    value: '2 / 9',
    detail: 'Partial strong10 run produced true generated-response flips.',
  },
  {
    tag: 'transfer bank',
    title: 'Heldout100 ASR',
    value: '75%',
    detail: 'Banked suffix transfer flips 75 incorrect samples to correct.',
  },
  {
    tag: 'ablation',
    title: 'No-bank ASR',
    value: '2%',
    detail: 'Removing suffix bank nearly eliminates correct-target flips.',
  },
];

const attackScenarios = {
  clean: {
    title: 'Clean grading',
    asr: 0,
    note: '没有注入 adversarial suffix。原始评分中 10 个样本已被误判为 correct，34 个为 contradictory，56 个为 incorrect。',
    dist: [
      { name: 'correct', value: 10, color: 'c-green' },
      { name: 'contradictory', value: 34, color: 'c-amber' },
      { name: 'incorrect', value: 56, color: 'c-red' },
    ],
  },
  nobank: {
    title: 'GCG without suffix bank',
    asr: 2,
    note: '只做当前样本的 screening suffix 优化，不使用迁移 suffix bank。攻击主要把 incorrect 推向 contradictory。',
    dist: [
      { name: 'correct', value: 12, color: 'c-green' },
      { name: 'contradictory', value: 46, color: 'c-amber' },
      { name: 'incorrect', value: 42, color: 'c-red' },
    ],
  },
  banked: {
    title: 'GCG with suffix bank',
    asr: 75,
    note: '使用成功 suffix bank 迁移后，大量错误答案被判为 correct，是当前攻击成功的主要来源。',
    dist: [
      { name: 'correct', value: 85, color: 'c-green' },
      { name: 'contradictory', value: 12, color: 'c-amber' },
      { name: 'incorrect', value: 3, color: 'c-red' },
    ],
  },
  hybrid: {
    title: 'Hybrid upper bound',
    asr: 90,
    note: '混合了更显式的 prompt-injection 变体，应作为上界展示，不等同于纯 GCG。',
    dist: [
      { name: 'correct', value: 100, color: 'c-green' },
      { name: 'contradictory', value: 0, color: 'c-amber' },
      { name: 'incorrect', value: 0, color: 'c-red' },
    ],
  },
};

const compareRuns = {
  banked: {
    label: 'with suffix bank',
    badge: 'ASR 75%',
    comment: 'Suffix bank makes the attack decisive: the largest mass moves into correct.',
    matrix: [
      ['True / Pred', 'correct', 'contradictory', 'incorrect'],
      ['correct', 0, 0, 0],
      ['contradictory', 0, 0, 0],
      ['incorrect', 85, 12, 3],
    ],
    dist: attackScenarios.banked.dist,
  },
  nobank: {
    label: 'without suffix bank',
    badge: 'ASR 2%',
    comment: 'Removing the bank does not remove all movement, but it mostly stops correct-target flips.',
    matrix: [
      ['True / Pred', 'correct', 'contradictory', 'incorrect'],
      ['correct', 0, 0, 0],
      ['contradictory', 0, 0, 0],
      ['incorrect', 12, 46, 42],
    ],
    dist: attackScenarios.nobank.dist,
  },
};

const defenses = {
  perplexity: {
    title: 'Perplexity Filter',
    stage: 'Pre-processing',
    description: '检测异常高困惑度的 suffix 或 prompt，适合拦截 token-level GCG 这类不自然字符串。',
    strength: 78,
    cost: 22,
    coverage: 68,
    takeaway: '优点是轻量；风险是自然语言化的攻击可能绕过。',
  },
  smooth: {
    title: 'SmoothLLM',
    stage: 'Inference-time voting',
    description: '对输入做随机字符扰动，多次生成后多数投票。对脆弱 suffix 攻击有破坏作用。',
    strength: 70,
    cost: 66,
    coverage: 74,
    takeaway: '防御面更广，但推理成本明显增加。',
  },
  reminder: {
    title: 'Self Reminder',
    stage: 'Prompt hardening',
    description: '在评分前追加安全评分规则，提醒模型忽略学生答案中的指令注入。',
    strength: 52,
    cost: 14,
    coverage: 48,
    takeaway: '实现简单，适合 baseline 防护，但对白盒 suffix 不一定够强。',
  },
  paraphrase: {
    title: 'Paraphrase Defense',
    stage: 'Input rewriting',
    description: '先改写输入再评分，目标是破坏 adversarial suffix 的 token 结构。',
    strength: 64,
    cost: 58,
    coverage: 62,
    takeaway: '能削弱 suffix 结构，但可能引入语义漂移，需要谨慎评估。',
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

function renderEvidence() {
  const grid = document.getElementById('metric-grid');
  grid.innerHTML = evidenceMetrics.map((m) => `
    <article class="metric-card">
      <span class="metric-tag">${m.tag}</span>
      <h4>${m.title}</h4>
      <div class="metric-value">${m.value}</div>
      <div class="metric-detail">${m.detail}</div>
    </article>
  `).join('');
}

function renderAttackScenario(key) {
  const scenario = attackScenarios[key];
  document.getElementById('attack-summary').innerHTML = `
    <strong>${scenario.title}</strong>
    <span>ASR: ${scenario.asr}%</span>
    <p>${scenario.note}</p>
  `;
  document.getElementById('attack-bars').innerHTML = scenario.dist.map((d) => barRow(d.name, d.value, d.color)).join('') +
    `<div class="bar-legend">${scenario.dist.map((d) => `<span><i class="bar-dot ${d.color}"></i>${d.name}</span>`).join('')}</div>`;
}

function renderMatrix(view) {
  const data = compareRuns[view];
  document.getElementById('matrix-title').textContent = `Attack confusion matrix: ${data.label}`;
  document.getElementById('matrix-badge').textContent = data.badge;
  document.getElementById('matrix-legend').textContent = view === 'banked' ? '85 / 12 / 3' : '12 / 46 / 42';
  document.getElementById('matrix-comment').textContent = data.comment;

  const table = document.getElementById('matrix-table');
  table.innerHTML = data.matrix.map((row, idx) => {
    if (idx === 0) {
      return `<div class="matrix-grid">${row.map((cell) => `<div class="matrix-cell matrix-headcell">${cell}</div>`).join('')}</div>`;
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

  document.getElementById('matrix-bar').innerHTML = data.dist.map((d) => barRow(d.name, d.value, d.color)).join('') +
    `<div class="bar-legend">${data.dist.map((d) => `<span><i class="bar-dot ${d.color}"></i>${d.name}</span>`).join('')}</div>`;
}

function renderDefense(key) {
  const d = defenses[key];
  document.getElementById('defense-card').innerHTML = `
    <span class="metric-tag">${d.stage}</span>
    <h4>${d.title}</h4>
    <p>${d.description}</p>
    <strong>${d.takeaway}</strong>
  `;
  document.getElementById('defense-meter').innerHTML = [
    ['Attack disruption', d.strength, 'c-green'],
    ['Compute / latency cost', d.cost, 'c-amber'],
    ['Coverage breadth', d.coverage, 'c-slate'],
  ].map(([label, value, color]) => barRow(label, value, color)).join('');
}

function bindControls() {
  document.getElementById('attack-select').addEventListener('change', (event) => {
    renderAttackScenario(event.target.value);
  });

  document.querySelectorAll('.toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      renderMatrix(btn.dataset.view);
    });
  });

  document.querySelectorAll('.defense-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.defense-tab').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      renderDefense(btn.dataset.defense);
    });
  });
}

renderEvidence();
renderAttackScenario('banked');
renderMatrix('banked');
renderDefense('perplexity');
bindControls();
