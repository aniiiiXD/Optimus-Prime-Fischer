const $ = (selector) => document.querySelector(selector);
const boardElement = $('#board');
const controls = $('#drop-controls');
const turnIndicator = $('#turn-indicator');
let state = null;
let busy = false;

function symbol(value) {
  return value === 1 ? 'X' : value === -1 ? 'O' : '';
}

function setBusy(value) {
  busy = value;
  $('#piece-select').disabled = value;
  $('#new-game').disabled = value;
  renderControls();
}

function renderControls() {
  controls.replaceChildren();
  const legal = new Set(state?.legal_columns ?? []);
  for (let col = 0; col < 7; col += 1) {
    const button = document.createElement('button');
    button.className = 'drop-button';
    button.type = 'button';
    button.textContent = `↓ ${col + 1}`;
    button.setAttribute('aria-label', `Drop ${state?.human_piece ?? 'your'} piece in column ${col + 1}`);
    button.disabled = busy || !state?.human_turn || !legal.has(col);
    button.addEventListener('click', () => playColumn(col));
    controls.append(button);
  }
}

function renderBoard() {
  boardElement.replaceChildren();
  const winning = new Set((state?.winning_line ?? []).map(([row, col]) => `${row},${col}`));
  for (let row = 0; row < 6; row += 1) {
    for (let col = 0; col < 7; col += 1) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.setAttribute('role', 'gridcell');
      const value = state?.board?.[row]?.[col] ?? 0;
      const piece = document.createElement('span');
      piece.className = `disc ${value === 1 ? 'x' : value === -1 ? 'o' : ''}${winning.has(`${row},${col}`) ? ' winning' : ''}`;
      piece.setAttribute('aria-label', value === 0 ? `Row ${row + 1}, column ${col + 1}, empty` : `Row ${row + 1}, column ${col + 1}, ${symbol(value)}`);
      cell.append(piece);
      boardElement.append(cell);
    }
  }
}

function renderChoices(decision) {
  const choices = $('#choices');
  choices.replaceChildren();
  if (!decision) {
    choices.innerHTML = '<div class="empty-choice"><span>—</span><p>Waiting for the bot’s next decision.</p></div>';
    $('#analysis-copy').textContent = 'The model’s last decision will appear here after the bot makes a move.';
    $('#forward-time').textContent = '—';
    $('#scan-time').textContent = '—';
    $('#state-value').textContent = '—';
    return;
  }
  const reason = decision.decision_reason.replaceAll('_', ' ');
  $('#analysis-copy').textContent = `Decision: ${reason}. Probabilities are the model policy; the safety pass checks immediate wins and losses.`;
  decision.choices.slice(0, 3).forEach((choice, index) => {
    const row = document.createElement('div');
    row.className = `choice${choice.selected ? ' selected' : ''}`;
    const rank = document.createElement('span');
    rank.className = 'rank';
    rank.textContent = `0${index + 1}`;
    const main = document.createElement('div');
    main.className = 'choice-main';
    const name = document.createElement('span');
    name.className = 'column-name';
    name.textContent = `Column ${choice.column + 1}${choice.selected ? ' · selected' : ''}`;
    const detail = document.createElement('span');
    detail.className = 'choice-meta';
    const safety = choice.safe_next_turn === true ? 'SAFE NEXT TURN' : choice.safe_next_turn === false ? `REPLIES: ${(choice.opponent_winning_replies ?? []).join(', ') || '—'}` : 'MODEL ONLY';
    detail.textContent = `${(choice.probability * 100).toFixed(1)}% policy · ${safety}`;
    main.append(name, detail);
    const q = document.createElement('span');
    q.className = 'choice-value';
    q.innerHTML = `${choice.q >= 0 ? '+' : ''}${choice.q.toFixed(2)}<small>Q EST.</small>`;
    row.append(rank, main, q);
    choices.append(row);
  });
  $('#forward-time').textContent = `${decision.latency_ms.toFixed(2)} ms`;
  $('#scan-time').textContent = `${decision.tactical_scan_ms.toFixed(2)} ms`;
  $('#state-value').textContent = `${decision.value >= 0 ? '+' : ''}${decision.value.toFixed(3)}`;
}

function render(nextState) {
  state = nextState;
  $('#human-label').textContent = state.human_piece;
  $('#bot-label').textContent = state.bot_piece;
  $('#move-count').textContent = `${String(state.move_count).padStart(2, '0')} / 42`;
  $('#piece-select').value = state.human_piece;
  $('#piece-select').disabled = busy;
  $('#turn-text').textContent = state.message;
  turnIndicator.dataset.status = state.status;
  turnIndicator.classList.toggle('error', false);
  $('#board-hint').textContent = state.status === 'human'
    ? `You are ${state.human_piece}. Choose a column; the piece falls to the lowest open slot.`
    : state.status === 'win' || state.status === 'draw'
      ? 'Start a new game whenever you are ready for another round.'
      : 'The Transformer is evaluating its legal moves…';
  renderBoard();
  renderControls();
  renderChoices(state.decision);
}

async function request(path, body) {
  const response = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function playColumn(column) {
  if (busy || !state?.human_turn) return;
  setBusy(true);
  try {
    render(await request('/api/move', { column: column + 1 }));
  } catch (error) {
    $('#turn-text').textContent = error.message;
    turnIndicator.classList.add('error');
  } finally {
    setBusy(false);
  }
}

async function newGame() {
  setBusy(true);
  $('#turn-text').textContent = 'Starting a new game…';
  try {
    render(await request('/api/new', { human_piece: $('#piece-select').value }));
  } catch (error) {
    $('#turn-text').textContent = error.message;
    turnIndicator.classList.add('error');
  } finally {
    setBusy(false);
  }
}

$('#new-game').addEventListener('click', newGame);
$('#piece-select').addEventListener('change', newGame);

request('/api/state').then(render).catch((error) => {
  $('#turn-text').textContent = `Can't reach local game: ${error.message}`;
  turnIndicator.classList.add('error');
});
