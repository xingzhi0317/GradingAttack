const state = {
  sampleId: LAB_DATA.samples[0].id,
  attackId: 'gcgBank',
  defenseId: 'none',
  promptMode: 'original',
};

const distributions = {
  none: { asr: 0, row: [10, 34, 56], dist: [['correct', 10, 'c-green'], ['contradictory', 34, 'c-amber'], ['incorrect', 56, 'c-red']] },
  gcgNoBank: { asr: 2, row: [12, 46, 42], dist: [['correct', 12, 'c-green'], ['contradictory', 46, 'c-amber'], ['incorrect', 42, 'c-red']] },
  gcgBank: { asr: 75, row: [85, 12, 3], dist: [['correct', 85, 'c-green'], ['contradictory', 12, 'c-amber'], ['incorrect', 3, 'c-red']] },
  strongGcg: { asr: 22, row: [2, 3, 4], dist: [['correct', 2, 'c-green'], ['contradictory', 3, 'c-amber'], ['incorrect', 4, 'c-red']] },
  hybrid: { asr: 90, row: [100, 0, 0], dist: [['correct', 100, 'c-green'], ['contradictory', 0, 'c-amber'], ['incorrect', 0, 'c-red']] },
};

function $(id) {
  return document.getElementById(id);
}

function currentSample() {
  return LAB_DATA.samples.find((sample) => sample.id === state.sampleId);
}

function parseVerdict(output) {
  if (output.includes('correct') && !output.includes('incorrect')) return 'correct';
  if (output.includes('contradictory')) return 'contradictory';
  if (output.includes('incorrect')) return 'incorrect';
  if (output.includes('rejected')) return 'rejected';
  return 'unknown';
}

function verdictClass(verdict) {
  return {
    correct: 'chip-correct',
    contradictory: 'chip-contradictory',
    incorrect: 'chip-incorrect',
    rejected: 'chip-rejected',
  }[verdict] || 'chip-neutral';
}

function fillTemplate(sample) {
  return LAB_DATA.promptTemplate
    .replace('{{question}}', sample.question)
    .replace('{{reference}}', sample.reference)
    .replace('{{student}}', sample.student);
}

function attackedPrompt(sample, attack) {
  const original = fillTemplate(sample);
  if (!attack.suffix) return original;
  return `${original}\n\n[Adversarial suffix appended to student answer]\n${attack.suffix}`;
}

function defendedPrompt(sample, attack, defense) {
  const prompt = attackedPrompt(sample, attack);
  if (state.defenseId === 'none') return prompt;
  return `[Defense: ${defense.label}]\n${defense.action}\n\n${prompt}`;
}

function outputFor(sample, attack, defense) {
  const attacked = sample.attacks[state.attackId];
  if (state.defenseId === 'none' || state.attackId === 'none') return attacked;
  if (defense.verdictShift === 'rejected') return '{"verdict": "rejected", "reason": "adversarial suffix detected"}';
  return `{"verdict": "${defense.verdictShift || parseVerdict(attacked)}"}`;
}

function barRow(label, value, color, total = 100, suffix = '') {
  const pct = Math.max(0, Math.min(100, (value / total) * 100));
  return `
    <div class="bar-row">
      <div class="bar-label"><strong>${label}</strong><span>${value}${suffix}</span></div>
      <div class="bar-track"><div class="bar-seg ${color}" style="width:${pct}%"></div></div>
    </div>
  `;
}

function renderControls() {
  $('sample-select').innerHTML = LAB_DATA.samples.map((sample) =>
    `<option value="${sample.id}">${sample.title}</option>`
  ).join('');
  $('attack-select').innerHTML = Object.entries(LAB_DATA.attacks).map(([id, attack]) =>
    `<option value="${id}">${attack.label}</option>`
  ).join('');
  $('defense-select').innerHTML = Object.entries(LAB_DATA.defenses).map(([id, defense]) =>
    `<option value="${id}">${defense.label}</option>`
  ).join('');

  $('sample-select').value = state.sampleId;
  $('attack-select').value = state.attackId;
  $('defense-select').value = state.defenseId;
}

function renderSample(sample) {
  $('ground-truth').textContent = `Ground truth: ${sample.groundTruth}`;
  $('ground-truth').className = `status-chip ${verdictClass(sample.groundTruth)}`;
  $('sample-view').innerHTML = `
    <div><strong>Question</strong><p>${sample.question}</p></div>
    <div><strong>Reference answer</strong><p>${sample.reference}</p></div>
    <div><strong>Student answer</strong><p>${sample.student}</p></div>
  `;
  $('sample-note').innerHTML = `<strong>Why this sample?</strong><p>${sample.note}</p>`;
}

function renderPrompt(sample, attack, defense) {
  const prompt = {
    original: fillTemplate(sample),
    attacked: attackedPrompt(sample, attack),
    defended: defendedPrompt(sample, attack, defense),
  }[state.promptMode];
  $('prompt-view').textContent = prompt;
}

function renderVerdicts(sample, attack, defense) {
  const cleanVerdict = parseVerdict(sample.cleanOutput);
  const attackedOutput = sample.attacks[state.attackId];
  const attackedVerdict = parseVerdict(attackedOutput);
  const defendedOutput = outputFor(sample, attack, defense);
  const defendedVerdict = parseVerdict(defendedOutput);

  $('verdict-flow').innerHTML = [
    ['Clean', cleanVerdict],
    ['After attack', attackedVerdict],
    ['After defense', defendedVerdict],
  ].map(([label, verdict]) => `
    <div class="verdict-step">
      <span>${label}</span>
      <strong class="${verdictClass(verdict)}">${verdict}</strong>
    </div>
  `).join('');

  $('output-view').innerHTML = `
    <div class="output-block"><span>Original output</span><code>${sample.cleanOutput}</code></div>
    <div class="output-block"><span>Attacked output</span><code>${attackedOutput}</code></div>
    <div class="output-block"><span>Defended output</span><code>${defendedOutput}</code></div>
  `;

  const attackSucceeded = cleanVerdict !== 'correct' && attackedVerdict === 'correct';
  const defenseRecovered = attackedVerdict === 'correct' && defendedVerdict !== 'correct';
  $('defense-effect').innerHTML = `
    <div class="effect-row"><span>Attack succeeded</span><strong>${attackSucceeded ? 'yes' : 'no'}</strong></div>
    <div class="effect-row"><span>Defense recovered</span><strong>${defenseRecovered ? 'yes' : 'not in this setting'}</strong></div>
    <div class="effect-row"><span>Defense cost</span><strong>${defense.cost}</strong></div>
    <p>${defense.effect}</p>
  `;
}

function renderMetrics() {
  const data = distributions[state.attackId] || distributions.none;
  $('asr-chip').textContent = `ASR ${data.asr}%`;
  $('asr-chip').className = `status-chip ${data.asr >= 50 ? 'chip-correct' : 'chip-incorrect'}`;
  $('attack-bars').innerHTML = data.dist.map(([label, value, color]) => barRow(label, value, color)).join('');

  const row = data.row;
  $('matrix-table').innerHTML = `
    <div class="matrix-grid">
      <div class="matrix-cell matrix-headcell">True / Pred</div>
      <div class="matrix-cell matrix-headcell">correct</div>
      <div class="matrix-cell matrix-headcell">contradictory</div>
      <div class="matrix-cell matrix-headcell">incorrect</div>
      <div class="matrix-cell matrix-rowhead">incorrect</div>
      <div class="matrix-cell heat-${Math.ceil(row[0] / 25)}"><span class="matrix-score">${row[0]}</span></div>
      <div class="matrix-cell heat-${Math.ceil(row[1] / 25)}"><span class="matrix-score">${row[1]}</span></div>
      <div class="matrix-cell heat-${Math.ceil(row[2] / 25)}"><span class="matrix-score">${row[2]}</span></div>
    </div>
  `;
}

function renderMethodText(attack, defense) {
  const attackSummary = `${attack.short}: ${attack.description} ${attack.metric}`;
  const defenseSummary = `${defense.label}: ${defense.action}`;
  document.documentElement.style.setProperty('--active-asr', `${attack.asr}%`);
  $('sample-note').innerHTML += `<hr><p><strong>Attack:</strong> ${attackSummary}</p><p><strong>Defense:</strong> ${defenseSummary}</p>`;
}

function renderAll() {
  const sample = currentSample();
  const attack = LAB_DATA.attacks[state.attackId];
  const defense = LAB_DATA.defenses[state.defenseId];
  renderSample(sample);
  renderPrompt(sample, attack, defense);
  renderVerdicts(sample, attack, defense);
  renderMetrics();
  renderMethodText(attack, defense);
}

function bindEvents() {
  $('sample-select').addEventListener('change', (event) => {
    state.sampleId = event.target.value;
    renderAll();
  });
  $('attack-select').addEventListener('change', (event) => {
    state.attackId = event.target.value;
    renderAll();
  });
  $('defense-select').addEventListener('change', (event) => {
    state.defenseId = event.target.value;
    renderAll();
  });
  document.querySelectorAll('.segment').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.segment').forEach((b) => b.classList.remove('active'));
      button.classList.add('active');
      state.promptMode = button.dataset.prompt;
      renderAll();
    });
  });
}

renderControls();
renderAll();
bindEvents();
