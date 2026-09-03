/* KnowYourRights — SSE client and renderer.
 *
 * No framework and no build step. The interesting parts:
 *  - a hand-rolled SSE reader over fetch(), because EventSource cannot POST a body
 *  - a small markdown renderer that escapes first, so nothing a crawled page said can
 *    become live HTML in the answer
 *  - [S1] markers turned into chips that scroll to and highlight their source card
 */
'use strict';

const $ = (sel) => document.querySelector(sel);
const thread = $('#thread');
const threadInner = $('#threadInner');
const sourcesPane = $('#sources');
const input = $('#input');
const sendBtn = $('#send');
const statEl = $('#stat');

const state = {
  sessionId: localStorage.getItem('kyr.session') || '',
  depth: 'auto',
  userState: localStorage.getItem('kyr.state') || '',
  busy: false,
  controller: null,
  sources: new Map(),
  answerEl: null,
  answerText: '',
  timelineSteps: new Map(),
  lastQuestion: '',
};

/* ── escaping and a very small markdown subset ─────────────────────────────────── */
function esc(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Everything is escaped before any markup is added, so answer text can never inject HTML.
function renderMarkdown(src) {
  const lines = String(src || '').split('\n');
  let html = '';
  let list = null;

  const inline = (s) => esc(s)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g,
             '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>')
    .replace(/\[([A-Z]{1,2}\d{1,2})\]/g,
             '<button class="cite" data-cite="$1" title="Show source $1">$1</button>');

  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { closeList(); continue; }

    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const h3 = line.match(/^#{2,4}\s+(.*)$/);

    if (h3) { closeList(); html += `<h3>${inline(h3[1])}</h3>`; }
    else if (ol) {
      if (list !== 'ol') { closeList(); html += '<ol>'; list = 'ol'; }
      html += `<li>${inline(ol[2])}</li>`;
    } else if (ul) {
      if (list !== 'ul') { closeList(); html += '<ul>'; list = 'ul'; }
      html += `<li>${inline(ul[1])}</li>`;
    } else {
      closeList();
      html += `<p>${inline(line)}</p>`;
    }
  }
  closeList();
  return html;
}

/* ── DOM helpers ───────────────────────────────────────────────────────────────── */
function el(tag, cls, html) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function atBottom() {
  return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
}
function scrollDown(force) {
  if (force || atBottom()) thread.scrollTop = thread.scrollHeight;
}

/* ── turn scaffolding ──────────────────────────────────────────────────────────── */
let turn = null;

function startTurn(question) {
  state.sources.clear();
  state.answerText = '';
  state.timelineSteps.clear();
  sourcesPane.innerHTML = '<p class="empty">Searching…</p>';

  const userMsg = el('div', 'msg user', `<div class="body">${esc(question)}</div>`);
  threadInner.appendChild(userMsg);

  const wrap = el('div', 'msg');
  const timeline = el('details', 'timeline');
  timeline.open = true;
  timeline.innerHTML = '<summary><span class="tl-label">Researching…</span></summary><div class="steps"></div>';
  const notices = el('div', 'notices');
  const answer = el('div', 'answer');
  wrap.append(timeline, notices, answer);
  threadInner.appendChild(wrap);

  turn = {
    wrap, timeline,
    steps: timeline.querySelector('.steps'),
    label: timeline.querySelector('.tl-label'),
    notices, answer,
    started: Date.now(),
  };
  state.answerEl = answer;
  scrollDown(true);
}

function setStep(id, label, status, detail, query) {
  if (!turn) return;
  let node = state.timelineSteps.get(id);
  if (!node) {
    node = el('div', 'step');
    node.innerHTML = '<span class="dot"></span><span class="label"></span>'
                   + '<span class="detail"></span>';
    state.timelineSteps.set(id, node);
    turn.steps.appendChild(node);
  }
  node.className = `step ${status || 'running'}`;
  node.querySelector('.label').textContent = label;
  const det = node.querySelector('.detail');
  det.innerHTML = query ? `<span class="q">${esc(query)}</span>` : esc(detail || '');
  scrollDown();
}

function addNotice(data) {
  if (!turn) return;
  const level = data.level === 'pause' ? 'pause' : (data.level === 'warn' ? 'warn' : '');
  const node = el('div', `notice ${level}`);
  const icon = level === 'pause' ? '⏳' : (level === 'warn' ? '⚠' : 'ℹ');
  node.innerHTML = `<span aria-hidden="true">${icon}</span><span class="text">${esc(data.text)}</span>`;
  turn.notices.appendChild(node);

  // A rate-limit pause is the one wait long enough to look like a hang. Counting down
  // out loud is the difference between "it's working" and "it's broken".
  if (data.resume_in_s > 1) {
    let left = Math.ceil(data.resume_in_s);
    const span = node.querySelector('.text');
    const base = data.text.replace(/\s*Waiting \d+s.*/, '');
    const tick = () => {
      if (left <= 0) { node.remove(); clearInterval(timer); return; }
      span.innerHTML = `${esc(base)} <span class="count">${left}s</span>`;
      left -= 1;
    };
    tick();
    const timer = setInterval(tick, 1000);
    node.dataset.timer = String(timer);
  }
  scrollDown();
}

function addSafety(data) {
  if (!turn) return;
  const chips = (data.helplines || [])
    .map((h) => `<span class="helpline">${esc(h.label)} <b>${esc(h.number)}</b></span>`).join('');
  const node = el('div', 'safety');
  node.innerHTML = `<h3>If you need help right now</h3><p>${esc(data.text)}</p>`
                 + `<div class="helplines">${chips}</div>`;
  turn.notices.prepend(node);
  scrollDown(true);
}

function addProcedure(data) {
  if (!turn) return;
  const steps = (data.steps || []).map((s) => `<li>${esc(s.text)}</li>`).join('');
  const facts = [
    ['Fee', data.fees], ['Time limit', data.timeline],
    ['Appeal to', data.appeal_to],
    ['Documents', (data.documents || []).join(', ')],
  ].filter(([, v]) => v && String(v).trim())
   .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');

  const node = el('div', 'procedure');
  node.innerHTML = `<h3>${esc(data.title || 'How to do this')}</h3>`
    + (steps ? `<ol>${steps}</ol>` : '')
    + (facts ? `<dl class="facts">${facts}</dl>` : '')
    + (data.portal_url
        ? `<p style="margin:10px 0 0"><a href="${esc(data.portal_url)}" target="_blank" rel="noopener noreferrer">Open the official portal →</a></p>`
        : '');
  turn.notices.appendChild(node);
  scrollDown();
}

/* ── sources panel ─────────────────────────────────────────────────────────────── */
const TIER_ORDER = { statute: 0, official: 1, 'legal portal': 2, background: 3, web: 4 };

function renderSources() {
  if (!state.sources.size) {
    sourcesPane.innerHTML = '<p class="empty">No strongly relevant source was found for this answer.</p>';
    return;
  }
  const items = [...state.sources.values()]
    .sort((a, b) => (TIER_ORDER[a.tier_label] ?? 9) - (TIER_ORDER[b.tier_label] ?? 9)
                 || (b.score - a.score));

  const userState = state.userState;

  sourcesPane.innerHTML = items.map((s) => {
    const badges = [];

    // Jurisdiction leads, always — for a legal answer it is the first thing that decides
    // whether a provision even applies to the reader.
    if (s.jurisdiction === 'CENTRAL') {
      badges.push('<span class="badge juris central">Central law · all India</span>');
    } else if (s.jurisdiction === 'CONSTITUTION') {
      badges.push('<span class="badge juris central">Constitution · all India</span>');
    } else if (s.jurisdiction === 'STATE') {
      const mismatch = userState && s.state &&
                       userState.toLowerCase() !== s.state.toLowerCase();
      badges.push(`<span class="badge juris ${mismatch ? 'mismatch' : 'state'}">`
        + `${esc(s.state)} only${mismatch ? ` — not ${esc(userState)}` : ''}</span>`);
    }

    if (s.status === 'in_force') badges.push('<span class="badge force">in force</span>');
    if (s.status === 'omitted') badges.push('<span class="badge omitted">omitted</span>');
    if (s.effective_date) badges.push(`<span class="badge">from ${esc(s.effective_date)}</span>`);
    if (s.category) badges.push(`<span class="badge">${esc(s.category)}</span>`);

    const title = s.url
      ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>`
      : esc(s.title);

    const mismatched = s.jurisdiction === 'STATE' && userState && s.state &&
                       userState.toLowerCase() !== s.state.toLowerCase();
    return `<div class="src${mismatched ? ' mismatch' : ''}" id="src-${esc(s.id)}"
                 data-kind="${esc(s.kind)}">
      <div class="top"><span class="id">${esc(s.id)}</span>
        <span class="tier">${esc(s.tier_label)}</span></div>
      <div class="title">${title}</div>
      ${s.domain ? `<div class="domain">${esc(s.domain)}</div>` : ''}
      <div class="snippet">${esc(s.snippet)}</div>
      ${badges.length ? `<div class="meta">${badges.join('')}</div>` : ''}
      ${mismatched ? `<div class="warn-line">This is ${esc(s.state)} law and does not
        apply in ${esc(userState)}.</div>` : ''}
    </div>`;
  }).join('');
}

function flashSource(id) {
  const node = document.getElementById(`src-${id}`);
  if (!node) return;
  node.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  node.classList.add('flash');
  setTimeout(() => node.classList.remove('flash'), 1400);
}

/* ── the stream ────────────────────────────────────────────────────────────────── */
async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  state.lastQuestion = question;
  sendBtn.classList.add('stop');
  sendBtn.title = 'Stop';
  startTurn(question);

  state.controller = new AbortController();
  let response;
  try {
    response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: question, session_id: state.sessionId,
        depth: state.depth, state: state.userState,
      }),
      signal: state.controller.signal,
    });
  } catch (err) {
    finishTurn(`Could not reach the server: ${err.message}`);
    return;
  }
  if (!response.ok || !response.body) {
    finishTurn(`Server returned ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop() || '';
      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        try { handleEvent(JSON.parse(line.slice(5).trim())); }
        catch { /* a malformed frame must not kill the stream */ }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') addNotice({ level: 'warn', text: `Stream ended: ${err.message}` });
  }
  finishTurn();
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'session':
      state.sessionId = ev.session_id;
      localStorage.setItem('kyr.session', ev.session_id);
      break;

    case 'plan':
      if (turn) {
        turn.label.textContent = ev.depth === 'deep'
          ? 'Researching in depth…'
          : (ev.depth === 'quick' ? 'Looking it up…' : 'Researching…');
      }
      break;

    case 'tool':
      setStep(`tool:${ev.tool}:${ev.query}`,
              ({ legal_db: 'Searching Indian law', web: 'Searching the web',
                 official: 'Checking official sources', wikipedia: 'Reading background',
                 navigate: 'Navigating the portal', crawl: 'Reading pages',
                 verify: 'Fact-checking my own answer' })[ev.tool] || ev.tool,
              ev.status,
              ev.status === 'done' ? `${ev.count} found · ${ev.elapsed_ms} ms` : '',
              ev.status === 'running' ? ev.query : '');
      break;

    case 'source':
      state.sources.set(ev.id, ev);
      renderSources();
      break;

    case 'sources_final':
      // Packing re-assigns ids and may add evidence recalled from earlier turns, so replace
      // rather than merge — otherwise a stale id lingers and its chip resolves to nothing.
      state.sources.clear();
      for (const s of ev.sources || []) state.sources.set(s.id, s);
      renderSources();
      break;

    case 'procedure': addProcedure(ev); break;
    case 'notice':    addNotice(ev); break;
    case 'safety':    addSafety(ev); break;

    case 'stage':
      // A rewrite after fact-checking must replace the draft, not append to it.
      if (ev.id === 'write' && ev.status === 'running' && state.answerText) {
        state.answerText = '';
        if (state.answerEl) state.answerEl.innerHTML = '';
      }
      setStep(ev.id, ev.label, ev.status, ev.detail);
      break;

    case 'token':
      state.answerText += ev.delta;
      if (state.answerEl) {
        state.answerEl.innerHTML = renderMarkdown(state.answerText) + '<span class="cursor"></span>';
        scrollDown();
      }
      break;

    case 'answer_revised':
      state.answerText = ev.text;
      if (state.answerEl) state.answerEl.innerHTML = renderMarkdown(state.answerText);
      break;

    case 'verdict':
      if (turn) {
        const bits = [];
        if (ev.citations_verified) bits.push(`${ev.citations_verified} citation(s) verified`);
        if (ev.unsupported?.length) bits.push(`${ev.unsupported.length} removed as unverifiable`);
        turn.verdict = bits.join(' · ');
      }
      break;

    case 'usage':
      statEl.textContent = [
        `${ev.elapsed_s}s`, `${ev.llm_calls} calls`,
        ev.crawls ? `${ev.crawls} pages read` : '',
        ev.throttled ? 'rate-limited' : '',
      ].filter(Boolean).join(' · ');
      break;

    case 'error':
      addNotice({ level: 'warn', text: ev.message });
      break;
  }
}

function finishTurn(errorText) {
  state.busy = false;
  state.controller = null;
  sendBtn.classList.remove('stop');
  sendBtn.title = 'Send (Enter)';

  if (turn) {
    turn.notices.querySelectorAll('.notice[data-timer]').forEach((n) => {
      clearInterval(Number(n.dataset.timer)); n.remove();
    });
    if (errorText) addNotice({ level: 'warn', text: errorText });
    if (state.answerEl) state.answerEl.innerHTML = renderMarkdown(state.answerText);

    const secs = ((Date.now() - turn.started) / 1000).toFixed(1);
    turn.label.textContent = `Research · ${secs}s`;
    turn.timeline.open = false;

    if (state.answerText.trim()) turn.wrap.appendChild(buildVerdict(turn.verdict));
    if (!state.sources.size) renderSources();
  }
  turn = null;
  input.focus();
}

function buildVerdict(text) {
  const row = el('div', 'verdict');
  row.innerHTML = `<span>${esc(text || '')}</span>`;
  const rate = el('div', 'rate');
  for (const [value, glyph, label] of [['up', '👍', 'Helpful'], ['down', '👎', 'Not helpful']]) {
    const button = el('button', null, glyph);
    button.setAttribute('aria-label', label);
    button.onclick = () => {
      rate.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', 'false'));
      button.setAttribute('aria-pressed', 'true');
      fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: state.sessionId, rating: value,
          question: state.lastQuestion, answer: state.answerText.slice(0, 4000),
        }),
      }).catch(() => {});
    };
    rate.appendChild(button);
  }
  const copy = el('button', null, 'Copy');
  copy.onclick = async () => {
    const cites = [...state.sources.values()].map((s) => `[${s.id}] ${s.title}${s.url ? ` — ${s.url}` : ''}`);
    await navigator.clipboard.writeText(`${state.answerText}\n\nSources:\n${cites.join('\n')}`);
    copy.textContent = 'Copied';
    setTimeout(() => { copy.textContent = 'Copy'; }, 1500);
  };
  rate.appendChild(copy);
  row.appendChild(rate);
  return row;
}

/* ── wiring ────────────────────────────────────────────────────────────────────── */
function send() {
  if (state.busy) {
    state.controller?.abort();
    fetch('/api/stop', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    }).catch(() => {});
    return;
  }
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';
  ask(text);
}

sendBtn.onclick = send;
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 168)}px`;
});

document.addEventListener('click', (e) => {
  const chip = e.target.closest('.cite');
  if (chip) { flashSource(chip.dataset.cite); return; }
  const example = e.target.closest('.example');
  if (example && !state.busy) { input.value = example.textContent.trim(); send(); }
});

document.querySelectorAll('.segmented button').forEach((button) => {
  button.onclick = () => {
    document.querySelectorAll('.segmented button')
      .forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    state.depth = button.dataset.depth;
  };
});

$('#state').addEventListener('change', (e) => {
  state.userState = e.target.value;
  localStorage.setItem('kyr.state', state.userState);
});

$('#theme').onclick = () => {
  const now = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = now;
  localStorage.setItem('kyr.theme', now);
};

$('#reset').onclick = async () => {
  await fetch('/api/reset', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: state.sessionId }),
  }).catch(() => {});
  threadInner.innerHTML = '';
  state.sources.clear();
  renderSources();
  statEl.textContent = '';
  input.focus();
};

/* ── boot ──────────────────────────────────────────────────────────────────────── */
(function boot() {
  const saved = localStorage.getItem('kyr.theme');
  if (saved) document.documentElement.dataset.theme = saved;
  else if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    document.documentElement.dataset.theme = 'dark';
  }

  fetch('/api/health').then((r) => r.json()).then((h) => {
    const select = $('#state');
    for (const name of h.states || []) {
      const option = el('option');
      option.value = name; option.textContent = name;
      select.appendChild(option);
    }
    if (state.userState) select.value = state.userState;
    if (h.disclaimer) $('#disclaimer').textContent = h.disclaimer;
    if (!h.ready) {
      statEl.textContent = 'loading models…';
      setTimeout(() => fetch('/api/health').then((r) => r.json())
        .then((h2) => { if (h2.ready) statEl.textContent = ''; }), 8000);
    }
  }).catch(() => {});

  // ── pipeline explainer ───────────────────────────────────────────────────────────────
  // Opened from the "?" beside the depth buttons. Closes on the X, on the backdrop, and on
  // Escape — a panel that traps you is worse than no panel.
  const pipeline = $('#pipeline');
  const pipelineBtn = $('#pipelineBtn');
  if (pipeline && pipelineBtn) {
    const setPipeline = (open) => {
      pipeline.hidden = !open;
      pipelineBtn.setAttribute('aria-expanded', String(open));
      if (open) $('#pipelineClose').focus();
      else pipelineBtn.focus();
    };
    pipelineBtn.addEventListener('click', () => setPipeline(pipeline.hidden));
    $('#pipelineClose').addEventListener('click', () => setPipeline(false));
    pipeline.addEventListener('click', (e) => { if (e.target === pipeline) setPipeline(false); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !pipeline.hidden) setPipeline(false);
    });
  }

  input.focus();
})();
