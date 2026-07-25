const attacks = {
  none: {
    label: 'No attack',
    asr: 0,
    note: 'clean grading',
    output: 'incorrect',
    title: 'Clean grading keeps the answer incorrect',
    copy: 'No suffix is appended, so the judge returns the expected incorrect verdict.',
  },
  gcgBank: {
    label: 'GCG + suffix bank',
    asr: 75,
    note: 'transfer suffix bank enabled',
    output: 'correct',
    title: 'Banked GCG flips the judge',
    copy: 'A transferred suffix changes this incorrect answer into a correct verdict.',
  },
  gcgNoBank: {
    label: 'GCG without bank',
    asr: 2,
    note: 'same setup, bank removed',
    output: 'incorrect',
    title: 'No-bank GCG usually fails to reach correct',
    copy: 'The suffix can disturb the model, but this setting rarely produces a true correct flip.',
  },
  hybrid: {
    label: 'Hybrid injection',
    asr: 90,
    note: 'upper-bound reference',
    output: 'correct',
    title: 'Hybrid injection is a stronger upper bound',
    copy: 'This mixes explicit prompt-injection behavior, so it should not be read as pure GCG.',
  },
};

const defenses = {
  none: {
    label: 'No defense',
    output: null,
    copy: 'The attacked prompt is sent directly to the judge.',
  },
  selfReminder: {
    label: 'Self Reminder',
    output: 'contradictory',
    copy: 'A safety instruction asks the judge to ignore injected grading commands.',
  },
  pplFilter: {
    label: 'PPL Filter',
    output: 'rejected',
    copy: 'The suffix is rejected before it reaches the judge.',
  },
  smoothLlm: {
    label: 'SmoothLLM',
    output: 'incorrect',
    copy: 'Randomized perturbation and voting break the brittle suffix behavior.',
  },
  paraphrase: {
    label: 'Paraphrase',
    output: 'incorrect',
    copy: 'Rewriting weakens the adversarial token pattern before grading.',
  },
};

const state = {
  attack: 'gcgBank',
  defense: 'none',
};

function verdictClass(verdict) {
  return {
    correct: 'is-correct',
    incorrect: 'is-incorrect',
    contradictory: 'is-contradictory',
    rejected: 'is-rejected',
  }[verdict] || '';
}

function renderOptions(id, data, value) {
  const select = document.getElementById(id);
  select.innerHTML = Object.entries(data)
    .map(([key, item]) => `<option value="${key}">${item.label}</option>`)
    .join('');
  select.value = value;
}

function render() {
  const attack = attacks[state.attack];
  const defense = defenses[state.defense];
  const clean = 'incorrect';
  const attacked = attack.output;
  const defended = state.defense === 'none' ? attacked : defense.output;
  const recovered = attacked === 'correct' && defended !== 'correct';

  document.getElementById('asr-value').textContent = `ASR ${attack.asr}%`;
  document.getElementById('asr-note').textContent = attack.note;
  document.getElementById('result-title').textContent = state.defense === 'none'
    ? attack.title
    : recovered ? 'Defense recovers the verdict' : 'Defense changes the output';
  document.getElementById('result-copy').textContent = state.defense === 'none'
    ? attack.copy
    : defense.copy;
  document.getElementById('result-chip').textContent = recovered
    ? 'defense recovers'
    : attacked === 'correct' ? 'attack succeeds' : 'attack limited';
  document.getElementById('result-chip').className = `result-chip ${recovered ? 'is-recovered' : verdictClass(attacked)}`;
  document.getElementById('output-view').textContent =
    `clean:    ${clean}\nattacked: ${attacked}\ndefended: ${defended}`;

  document.getElementById('verdict-flow').innerHTML = [
    ['Clean', clean],
    ['Attack', attacked],
    ['Defense', defended],
  ].map(([label, verdict]) => `
    <div class="verdict-step">
      <span>${label}</span>
      <strong class="${verdictClass(verdict)}">${verdict}</strong>
    </div>
  `).join('');
}

renderOptions('attack-select', attacks, state.attack);
renderOptions('defense-select', defenses, state.defense);
render();

document.getElementById('attack-select').addEventListener('change', (event) => {
  state.attack = event.target.value;
  render();
});

document.getElementById('defense-select').addEventListener('change', (event) => {
  state.defense = event.target.value;
  render();
});
