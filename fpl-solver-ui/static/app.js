/* Solver console — no framework, no build step. */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const fmt = (v, dp = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(dp);

const bytes = n =>
  n < 1024 ? `${n} B`
  : n < 1048576 ? `${(n / 1024).toFixed(0)} KB`
  : `${(n / 1048576).toFixed(1)} MB`;

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

const squadCache = { list: [], loaded: false };

async function loadSquads() {
  try {
    const out = await api('/api/squads');
    squadCache.list = out.squads || [];
  } catch { squadCache.list = []; }
  squadCache.loaded = true;
  $$('select[data-squad-fill]').forEach(fillSquadOptions);
}

/* Fifteen IDs typed by hand before every solve is slow, and a mistyped ID is
   a valid solve of the wrong problem — it fails silently. Every option here
   is something an earlier step already produced. */
function fillSquadOptions(sel) {
  const keep = sel.value;
  sel.innerHTML = '';
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = squadCache.list.length ? 'Fill from…' : 'no squads found yet';
  sel.append(blank);
  squadCache.list.forEach(sq => {
    const o = document.createElement('option');
    o.value = sq.key;
    o.textContent = sq.label;
    o.title = sq.note || '';
    sel.append(o);
  });
  sel.value = keep;
  sel.disabled = !squadCache.list.length;
}

const state = { invariants: {}, settings: {}, dirty: false, plans: [], active: null };

/* ------------------------------------------------------------ navigation */

$$('.rail-item').forEach(btn => {
  btn.onclick = () => {
    $$('.rail-item').forEach(b => b.classList.toggle('is-active', b === btn));
    $$('.panel').forEach(p =>
      p.classList.toggle('is-active', p.id === `panel-${btn.dataset.panel}`));
    if (btn.dataset.panel === 'plans' || btn.dataset.panel === 'compare') loadPlans();
  };
});

/* ------------------------------------------------------------ 01 sources */

const dz = $('#dropzone');
const fileInput = $('#file-input');

dz.onclick = () => fileInput.click();
fileInput.onchange = () => { upload(fileInput.files); fileInput.value = ''; };

['dragenter', 'dragover'].forEach(ev =>
  dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('is-over'); }));
['dragleave', 'drop'].forEach(ev =>
  dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('is-over'); }));
dz.addEventListener('drop', e => upload(e.dataTransfer.files));

async function upload(fileList) {
  if (!fileList || !fileList.length) return;
  const form = new FormData();
  [...fileList].forEach(f => form.append('files', f));
  const report = $('#upload-report');
  report.hidden = false;
  report.className = 'notice';
  report.textContent = `Uploading ${fileList.length} file${fileList.length > 1 ? 's' : ''}…`;
  try {
    const out = await api('/api/upload', { method: 'POST', body: form });
    renderFiles(out.files);
    if (out.rejected.length) {
      report.className = 'notice notice-warn';
      report.textContent = out.rejected.map(r => `${r.name} — ${r.why}`).join('  ');
    } else {
      report.className = 'notice';
      report.textContent = `Saved ${out.saved.join(', ')}.`;
      setTimeout(() => { report.hidden = true; }, 4000);
    }
  } catch (err) {
    report.className = 'notice notice-bad';
    report.textContent = err.message;
  }
}

function renderFiles(files) {
  const body = $('#file-table tbody');
  body.textContent = '';
  $('#files-empty').hidden = files.length > 0;
  $('#file-table').hidden = files.length === 0;

  files.forEach(f => {
    const tr = el('tr');
    tr.append(el('td', 'name', f.name));

    const kind = el('td');
    kind.append(el('span', 'tag', f.kind));
    tr.append(kind);

    tr.append(el('td', 'num', f.rows === null ? '–' : f.rows.toLocaleString()));
    tr.append(el('td', 'num', bytes(f.size)));
    tr.append(el('td', 'num', new Date(f.modified).toLocaleString([], {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })));

    const act = el('td', 'num');
    const rm = el('button', 'btn btn-ghost btn-sm', 'Remove');
    rm.onclick = async () => {
      if (!confirm(`Remove ${f.name} from the data folder?`)) return;
      renderFiles((await api(`/api/files/${encodeURIComponent(f.name)}`,
        { method: 'DELETE' })).files);
    };
    act.append(rm);
    tr.append(act);
    body.append(tr);
  });
}

/* ------------------------------------------------------------ 02 settings */

$$('.seg[data-mode]').forEach(b => {
  b.onclick = () => {
    $$('.seg[data-mode]').forEach(x => x.classList.toggle('is-active', x === b));
    const raw = b.dataset.mode === 'raw';
    if (raw) $('#settings-raw').value = JSON.stringify(collectSettings(), null, 2);
    $('#settings-raw').hidden = !raw;
    $('#settings-form').hidden = raw;
  };
});

async function loadSettings() {
  const out = await api('/api/settings');
  state.settings = out.settings;
  $('#settings-raw').value = out.raw;
  renderSettingsForm(out.settings);
  if (!out.exists) flash('#settings-warnings', 'notice notice-warn',
    'No settings file yet. Saving will create one at the path above.');
}

function renderSettingsForm(settings) {
  const grid = $('#settings-form');
  grid.textContent = '';

  Object.entries(settings).forEach(([key, value]) => {
    const locked = Object.prototype.hasOwnProperty.call(state.invariants, key);
    const field = el('div', 'field' + (locked ? ' is-locked' : ''));
    field.dataset.key = key;

    const top = el('div', 'field-top');
    top.append(el('label', null, key));
    if (locked) top.append(el('span', 'lock', 'INVARIANT'));
    field.append(top);

    if (typeof value === 'boolean') {
      const wrap = el('label', 'switch');
      const input = el('input');
      input.type = 'checkbox';
      input.checked = value;
      wrap.append(input, el('span', null, value ? 'on' : 'off'));
      input.onchange = () => {
        wrap.lastChild.textContent = input.checked ? 'on' : 'off';
        markDirty();
      };
      field.dataset.type = 'bool';
      field.append(wrap);
    } else if (typeof value === 'number') {
      const input = el('input');
      input.type = 'number';
      input.step = Number.isInteger(value) ? '1' : 'any';
      input.value = value;
      input.oninput = markDirty;
      field.dataset.type = 'number';
      field.append(input);
      if (locked) field.append(el('div', 'hint',
        `Expected ${state.invariants[key]} — solver rankings are meaningless above zero gap.`));
    } else if (value === null || typeof value === 'string') {
      const input = el('input');
      input.type = 'text';
      input.value = value ?? '';
      input.oninput = markDirty;
      field.dataset.type = 'string';
      field.append(input);
    } else {
      const ta = el('textarea');
      ta.value = JSON.stringify(value, null, 2);
      ta.oninput = markDirty;
      field.dataset.type = 'json';
      field.append(ta);
      field.append(el('div', 'hint', Array.isArray(value)
        ? `list · ${value.length} item${value.length === 1 ? '' : 's'}` : 'object'));
    }
    grid.append(field);
  });
}

function collectSettings() {
  if (!$('#settings-raw').hidden) {
    try { return JSON.parse($('#settings-raw').value); }
    catch (e) { throw new Error(`Raw JSON won't parse: ${e.message}`); }
  }
  const out = {};
  $$('.field', $('#settings-form')).forEach(f => {
    const key = f.dataset.key;
    const input = $('input, textarea', f);
    switch (f.dataset.type) {
      case 'bool':   out[key] = input.checked; break;
      case 'number': out[key] = input.value === '' ? null : Number(input.value); break;
      case 'json':
        try { out[key] = JSON.parse(input.value); }
        catch (e) { throw new Error(`'${key}' isn't valid JSON: ${e.message}`); }
        break;
      default:       out[key] = input.value;
    }
  });
  return out;
}

function markDirty() { state.dirty = true; $('#saved-note').hidden = true; }

$('#settings-save').onclick = async () => {
  try {
    const out = await api('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings: collectSettings() })
    });
    state.dirty = false;
    const note = $('#saved-note');
    note.hidden = false;
    note.textContent = `Saved ${new Date().toLocaleTimeString()}`;
    if (out.warnings.length) {
      flash('#settings-warnings', 'notice notice-warn', out.warnings.join('  '));
    } else {
      $('#settings-warnings').hidden = true;
    }
  } catch (err) {
    flash('#settings-warnings', 'notice notice-bad', err.message);
  }
};

$('#settings-reload').onclick = () => {
  if (state.dirty && !confirm('Discard unsaved changes and reload from disk?')) return;
  state.dirty = false;
  $('#settings-warnings').hidden = true;
  $('#saved-note').hidden = true;
  loadSettings();
};

function flash(sel, cls, text) {
  const node = $(sel);
  node.className = cls;
  node.textContent = text;
  node.hidden = false;
}

/* ------------------------------------------------------------ 03 run */

function renderSteps(steps) {
  const wrap = $('#steps');
  wrap.textContent = '';

  steps.forEach(step => {
    const card = el('div', 'step' + (step.enabled === false ? ' is-off' : ''));
    card.id = `step-${step.id}`;
    card.dataset.status = step.last_run ? step.last_run.status : 'idle';

    const head = el('div', 'step-head');
    head.append(el('div', 'step-idx', step.index));

    const body = el('div', 'step-body');
    const title = el('div', 'step-title');
    title.append(el('h3', null, step.name));
    if (step.gate) title.append(el('span', 'gate-tag', 'GATE'));
    if (step.enabled === false) title.append(el('span', 'tag', 'skipped in Run all'));
    const status = el('span', 'status', step.last_run ? step.last_run.status : 'not run');
    status.dataset.s = step.last_run ? step.last_run.status : 'idle';
    title.append(status);
    body.append(title);
    body.append(el('p', 'step-blurb', step.blurb));
    if (step.params && step.params.length) {
      const row = el('div', 'params');
      step.params.forEach(p => {
        const f = el('label', 'param' + (p.width === 'wide' ? ' is-wide' : ''));
        f.append(el('span', 'param-label', p.label + (p.required ? ' *' : '')));
        const input = el('input');
        input.type = 'text';
        input.dataset.param = p.name;
        input.placeholder = p.placeholder || '';
        if (p.name === 'squad') {
          const pick = document.createElement('select');
          pick.className = 'squad-fill';
          pick.dataset.squadFill = '1';
          pick.onchange = () => {
            const sq = squadCache.list.find(s => s.key === pick.value);
            if (sq) { input.value = sq.ids.join(','); input.dispatchEvent(new Event('change')); }
            pick.value = '';
          };
          field.append(pick);
          fillSquadOptions(pick);
        }
        input.value = (step.saved_params && step.saved_params[p.name])
          ?? (state.settingsDefaults && state.settingsDefaults[p.name])
          ?? p.default ?? '';
        if (p.required) input.required = true;
        f.append(input);
        row.append(f);
      });
      body.append(row);
    }
    body.append(el('div', 'step-cmd', step.command));
    head.append(body);

    const actions = el('div', 'step-actions');
    const run = el('button', 'btn btn-sm', 'Run');
    run.onclick = () => startRun(step.id);
    actions.append(run);
    head.append(actions);
    card.append(head);

    const console_ = el('pre', 'console');
    console_.hidden = true;
    card.append(console_);

    if (step.last_run && step.last_run.tail.length) {
      console_.hidden = false;
      step.last_run.tail.forEach(l => console_.append(line(l)));
    }
    wrap.append(card);
    if (step.last_run && step.last_run.status === 'running') {
      attachStream(step.id, step.last_run.id);
    }
  });
}

function line(text, kind = 'line') {
  const span = el('span', kind === 'meta' ? 'l-meta' : /error|traceback|failed/i.test(text)
    ? 'l-bad' : '');
  span.textContent = text + '\n';
  return span;
}

function setStopButton(stepId, runId) {
  const card = $(`#step-${stepId}`);
  const btn = $('.step-actions .btn', card);
  const console_ = $('.console', card);
  btn.textContent = 'Stop';
  btn.disabled = false;
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = 'Stopping\u2026';
    try {
      // A solving process ignores SIGTERM until HiGHS returns from its C
      // call, so the server escalates to SIGKILL. Report which happened
      // rather than swallowing it — silence is what made this look broken.
      const out = await api(`/api/cancel/${runId}`, { method: 'POST' });
      console_.append(line(`# cancel: ${out.how}`, 'meta'));
    } catch (err) {
      console_.append(line(`# cancel failed: ${err.message}`, 'meta'));
      btn.disabled = false;
      btn.textContent = 'Stop';
    }
  };
}

/* Re-attach to a run still in flight after a page load. Without this a
   refresh orphans it: the solve keeps running server-side but the card shows
   a plain Run button, so the next click starts a second one. */
function attachStream(stepId, runId) {
  const card = $(`#step-${stepId}`);
  const console_ = $('.console', card);
  const status = $('.status', card);
  const btn = $('.step-actions .btn', card);

  console_.hidden = false;
  card.dataset.status = 'running';
  status.dataset.s = 'running';
  status.textContent = 'running';
  setStopButton(stepId, runId);

  const src = new EventSource(`/api/stream/${runId}`);
  src.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.kind === 'end') {
      src.close();
      card.dataset.status = msg.text;
      status.dataset.s = msg.text;
      status.textContent = msg.text;
      btn.disabled = false;
      btn.textContent = 'Run again';
      btn.onclick = () => startRun(stepId);
      if (msg.text === 'ok') { loadPlans(); loadSquads(); }
      return;
    }
    const atBottom = console_.scrollHeight - console_.scrollTop - console_.clientHeight < 40;
    console_.append(line(msg.text, msg.kind));
    if (atBottom) console_.scrollTop = console_.scrollHeight;
  };
  src.onerror = () => src.close();
}

async function startRun(stepId) {
  const card = $(`#step-${stepId}`);
  const console_ = $('.console', card);
  const status = $('.status', card);
  const btn = $('.step-actions .btn', card);

  console_.hidden = false;
  console_.textContent = '';
  card.dataset.status = 'running';
  status.dataset.s = 'running';
  status.textContent = 'running';

  const params = {};
  $$('input[data-param]', card).forEach(i => { params[i.dataset.param] = i.value; });

  let runId;
  try {
    const started = await api(`/api/run/${stepId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params })
    });
    runId = started.run_id;
    console_.append(line(`$ ${started.command}`, 'meta'));
  } catch (err) {
    card.dataset.status = 'failed';
    status.dataset.s = 'failed';
    status.textContent = 'failed';
    console_.append(line(err.message));
    return;
  }

  attachStream(stepId, runId);
}

$('#run-all').onclick = async () => {
  const btn = $('#run-all');
  btn.disabled = true;
  btn.textContent = 'Running…';
  try {
    const out = await api('/api/run-all', { method: 'POST' });
    $$('.rail-item').find(b => b.dataset.panel === 'run').click();
    for (const id of out.steps) {
      const done = await watchUntilDone(id);
      if (done !== 'ok') break;
    }
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run all steps';
    loadPlans();
  loadSquads();
  }
};

/* Run all is driven server-side; the UI polls each step's own run and streams
   it into the matching card so the output lands where you'd expect it. */
async function watchUntilDone(stepId) {
  for (let i = 0; i < 3600; i++) {
    const st = await api('/api/state');
    const step = st.steps.find(s => s.id === stepId);
    const card = $(`#step-${stepId}`);
    if (step && step.last_run && card) {
      const console_ = $('.console', card);
      const status = $('.status', card);
      console_.hidden = false;
      console_.textContent = '';
      step.last_run.tail.forEach(l => console_.append(line(l)));
      console_.scrollTop = console_.scrollHeight;
      card.dataset.status = step.last_run.status;
      status.dataset.s = step.last_run.status;
      status.textContent = step.last_run.status;
      if (step.last_run.status !== 'running') return step.last_run.status;
    }
    await new Promise(r => setTimeout(r, 900));
  }
  return 'running';
}

/* ------------------------------------------------------------ 04 plans */

$('#plans-refresh').onclick = loadPlans;
$('#show-squad').onchange = () => drawLadder();

async function loadPlans() {
  const out = await api('/api/plans');
  state.plans = out.plans;
  state.comparison = out.comparison;

  const hasPlans = out.plans.length > 0;
  $('#plans-empty').hidden = hasPlans;
  $('#compare-empty').hidden = hasPlans;

  const sw = $('#plan-switch');
  sw.textContent = '';
  out.plans.forEach(p => {
    const b = el('button', 'seg', p.label);
    b.dataset.variant = p.variant;
    b.onclick = () => { state.active = p.variant; drawLadder(); };
    sw.append(b);
  });
  if (!state.active || !out.plans.some(p => p.variant === state.active)) {
    state.active = hasPlans ? out.plans[0].variant : null;
  }
  drawLadder();
  drawComparison(out.comparison);

  const tied = out.comparison.tied_count || 0;
  const flag = $('#tied-flag');
  flag.hidden = tied < 2;
  flag.textContent = `${tied} plans inside the noise band`;
}

function drawLadder() {
  const wrap = $('#ladder');
  wrap.textContent = '';
  const plan = state.plans.find(p => p.variant === state.active);
  $$('#plan-switch .seg').forEach(b =>
    b.classList.toggle('is-active', b.dataset.variant === state.active));
  if (!plan) return;

  const showSquad = $('#show-squad').checked;
  const ladder = el('div', 'ladder');

  plan.gameweeks.forEach(gw => {
    const col = el('div', 'gw' + (gw.chip ? ' has-chip' : ''));
    if (gw.chip) col.append(el('div', 'gw-chip', gw.chip));

    const head = el('div', 'gw-head');
    head.append(el('div', 'gw-no', `GW${gw.gw}`));
    head.append(el('div', 'gw-xp', fmt(gw.xpts, 1)));
    col.append(head);

    const meta = el('div', 'gw-meta');
    meta.append(el('span', null, `ITB ${fmt(gw.itb, 1)}`));
    meta.append(el('span', null, `FT ${gw.ft ?? '–'}`));
    if (gw.hits) meta.append(el('span', 'hit', `−${gw.hits * 4}`));
    col.append(meta);

    const moves = gw.transfers || [];
    if (!moves.length) {
      col.append(el('div', 'move-hold', 'roll the transfer'));
    } else {
      moves.forEach(t => {
        const m = el('div', 'move');
        if (t.out) m.append(playerRow(t.out, 'move-out'));
        m.append(el('div', 'move-arrow'));
        if (t.in) m.append(playerRow(t.in, 'move-in'));
        col.append(m);
      });
    }

    if (showSquad && gw.squad && gw.squad.length) {
      const xi = el('div', 'xi');
      xi.append(el('div', 'xi-label', 'Eleven'));
      const order = { GKP: 0, DEF: 1, MID: 2, FWD: 3 };
      [...gw.squad]
        .sort((a, b) =>
          (a.starting === b.starting ? 0 : a.starting ? -1 : 1) ||
          (order[a.pos] ?? 9) - (order[b.pos] ?? 9) ||
          (b.xpts ?? 0) - (a.xpts ?? 0))
        .forEach(p => {
          const row = el('div', 'xi-row' + (p.starting ? '' : ' is-bench'));
          row.append(el('span', 'xi-pos', p.pos));
          const name = el('span', 'xi-name', p.name);
          row.append(name);
          if (p.captain) row.append(el('span', 'armband', 'C'));
          else if (p.vice) row.append(el('span', 'armband v', 'V'));
          row.append(el('span', 'xi-xp', fmt(p.xpts, 1)));
          xi.append(row);
        });
      col.append(xi);
    }
    ladder.append(col);
  });

  wrap.append(ladder);
}

function playerRow(p, cls) {
  const row = el('div', `move-row ${cls}`);
  const left = el('div');
  left.append(el('div', 'move-name', p.name));
  if (p.team || p.pos) left.append(el('div', 'move-team', `${p.pos || ''} ${p.team || ''}`.trim()));
  row.append(left);
  row.append(el('div', 'move-price', fmt(p.price, 1)));
  return row;
}

/* ------------------------------------------------------------ 05 compare */

function drawComparison(cmp) {
  const wrap = $('#compare-wrap');
  wrap.textContent = '';
  if (!cmp || !cmp.rows.length) return;

  const note = el('div', 'cmp-note');
  note.append(el('i'));
  note.append(el('span', null,
    `Shaded rows finish within ${fmt(cmp.noise_band, 1)} xPts of the leader.`));
  wrap.append(note);

  const scroll = el('div', 'cmp-scroll');
  const table = el('table', 'cmp');

  const thead = el('thead');
  const hr = el('tr');
  hr.append(el('th', null, 'Variant'));
  cmp.gameweeks.forEach(gw => hr.append(el('th', null, `GW${gw}`)));
  ['Total', 'Hits', 'Moves', 'Gap'].forEach(h => hr.append(el('th', null, h)));
  thead.append(hr);
  table.append(thead);

  const bestPerGw = cmp.gameweeks.map((_, i) =>
    Math.max(...cmp.rows.map(r => r.cells[i] ?? -Infinity)));

  const tbody = el('tbody');
  cmp.rows.forEach(r => {
    const tr = el('tr', r.tied ? 'is-tied' : '');
    tr.append(el('td', null, r.label));
    r.cells.forEach((v, i) => {
      const td = el('td', v !== null && v === bestPerGw[i] ? 'cell-best' : '', fmt(v, 1));
      tr.append(td);
    });
    tr.append(el('td', 'total', fmt(r.total, 1)));
    tr.append(el('td', null, r.hits ? `−${r.hits * 4}` : '0'));
    tr.append(el('td', null, String(r.transfers)));
    tr.append(el('td',
      r.leader ? 'gap-lead' : r.tied ? 'gap-tied' : 'gap-real',
      r.leader ? 'leader' : `−${fmt(r.gap, 2)}`));
    tbody.append(tr);
  });
  table.append(tbody);
  scroll.append(table);
  wrap.append(scroll);

  const tied = cmp.tied_count;
  wrap.append(el('div', 'cmp-legend', tied > 1
    ? `${tied} of ${cmp.rows.length} plans finish inside the band. Treat their ordering as a tie and pick on structure — fixture cover, price rises, how much the plan commits you to — rather than on the decimal.`
    : 'One plan sits clear of the band. That gap is large enough to act on.'));
}

/* ------------------------------------------------------------ boot */

(async function boot() {
  try {
    const st = await api('/api/state');
    state.invariants = st.invariants;
    state.settingsDefaults = st.settings_defaults || {};
    $('#paths').textContent = `${st.paths.root}  ·  data/  ·  results/`;
    $('#settings-path').textContent = st.paths.settings;
    renderFiles(st.files);
    renderSteps(st.steps);
    const note = $('#archive-note');
    if (!st.archive) {
      note.hidden = false;
      note.textContent = `No snapshot found under ${st.paths.archive_root}. Steps that need `
        + `{ARCHIVE} will fail until the archive step has run at least once, or set `
        + `FPL_ARCHIVE_DIR to where your snapshots actually live.`;
    }
    await loadSettings();
    await loadPlans();
  } catch (err) {
    document.body.prepend(Object.assign(el('div', 'notice notice-bad',
      `Couldn't reach the server: ${err.message}`), { style: 'margin:16px' }));
  }
})();

window.addEventListener('beforeunload', e => {
  if (state.dirty) { e.preventDefault(); e.returnValue = ''; }
});
