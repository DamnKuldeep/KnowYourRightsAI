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
const app = $('.app');
const sourcesBtn = $('#sourcesBtn');
const sourcesCount = $('#sourcesCount');
const input = $('#input');
const sendBtn = $('#send');
const statEl = $('#stat');
const quotaEl = $('#quota');
const welcome = $('#welcome');
const SOURCES_EMPTY = sourcesPane.innerHTML;

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
  locked: false,        // a usage limit was reached; no more questions from this page
  config: {},           // /api/config: budgets, whether a reset code is accepted, the user
  quota: null,          // the last /api/quota answer
};

/* ── escaping and a very small markdown subset ─────────────────────────────────── */
function esc(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Escaping stops markup, not a "javascript:" link. Only web addresses become links.
function safeUrl(url) {
  return /^https?:\/\//i.test(String(url || '')) ? esc(url) : '';
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
    // A bare URL at the end of a sentence took the full stop with it — "rtionline.gov.in/."
    // linked to a page that does not exist. Trailing punctuation stays outside the link.
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, (_, pre, url) => {
      const tail = (url.match(/[.,;:!?]+$/) || [''])[0];
      const href = tail ? url.slice(0, -tail.length) : url;
      return `${pre}<a href="${href}" target="_blank" rel="noopener noreferrer">${href}</a>${tail}`;
    })
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

/* ── suggestions ───────────────────────────────────────────────────────────────── */
// Ideas for a first question, one per topic so a set never repeats a subject. Each is answerable
// from the corpus; the state-law ones show how jurisdiction is handled.
const SUGGESTIONS = {
  'Police & arrest': [
    'Can the police arrest me without a warrant?',
    'Do I have to unlock my phone if the police ask me to?',
    'Police ne FIR likhne se mana kar diya, ab kya karun?',
  ],
  'Work & pay': [
    "My employer hasn't paid my salary for two months. What can I do?",
    'How much maternity leave am I entitled to?',
    'Do I get gratuity if I resign after five years?',
  ],
  'Renting': [
    "My landlord won't return my security deposit",
    'Can my landlord evict me without notice?',
  ],
  'Shopping & services': [
    "An online seller won't refund a defective product",
    'How do I file a consumer complaint, and what does it cost?',
  ],
  'Government': [
    'How do I file an RTI, and what does it cost?',
    'My RTI has not been answered in 30 days. What next?',
  ],
  'Online & money': [
    'Someone cheated me online. Which law makes that a crime?',
    'Someone is sharing my photos online without consent',
    "A cheque I was given has bounced. What are my options?",
  ],
  'Family': [
    "What are a daughter's rights in her father's property?",
    'दहेज माँगना क्या अपराध है?',
    'How is maintenance decided after a divorce?',
  ],
  'Constitution': [
    'What does Article 21 of the Constitution protect?',
    'Can I be punished twice for the same offence?',
  ],
};

const pick = (list) => list[Math.floor(Math.random() * list.length)];

function shuffled(list) {                 // Fisher–Yates; sort() with a random key is biased
  const out = [...list];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function renderSuggestions(count = 6) {
  const topics = shuffled(Object.keys(SUGGESTIONS)).slice(0, count);
  $('#suggestions').replaceChildren(...topics.map((topic) => {
    const card = el('button', 'suggestion');
    card.type = 'button';
    card.innerHTML = `<span class="s-topic">${esc(topic)}</span>`
                   + `<span class="s-q">${esc(pick(SUGGESTIONS[topic]))}</span>`;
    return card;
  }));
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
  welcome.remove();                 // the empty state has done its job
  state.sources.clear();
  state.answerText = '';
  state.timelineSteps.clear();
  sourcesPane.innerHTML = '<p class="empty">Searching…</p>';
  setSourceCount(0);

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

// The card carries the *facts* — fee, time limit, appeal, documents, portal — and not the steps.
// Every writer model measured writes the steps itself, as a cited numbered list, whatever it is
// told; showing them here as well made each procedure read twice. The answer owns the steps and
// the citations; the card is the at-a-glance summary beside it.
function addProcedure(data) {
  if (!turn) return;
  const facts = [
    ['Fee', data.fees], ['Time limit', data.timeline],
    ['Appeal to', data.appeal_to],
    ['Documents', (data.documents || []).join(', ')],
  ].filter(([, v]) => v && String(v).trim())
   .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');

  const portal = safeUrl(data.portal_url);
  if (!facts && !portal) return;                // nothing worth a card of its own
  const node = el('div', 'procedure');
  node.innerHTML = `<h3>At a glance</h3>`
    + (facts ? `<dl class="facts">${facts}</dl>` : '')
    + (portal
        ? `<p class="portal"><a href="${portal}" target="_blank" rel="noopener noreferrer">Open the official portal →</a></p>`
        : '');
  turn.notices.appendChild(node);
  scrollDown();
}

/* ── sources panel ─────────────────────────────────────────────────────────────── */
const TIER_ORDER = { statute: 0, official: 1, 'legal portal': 2, background: 3, web: 4 };

function renderSources() {
  setSourceCount(state.sources.size);
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
    } else if (s.jurisdiction === 'STATE' || s.jurisdiction === 'TERRITORY') {
      // A Union Territory Act was passed by Parliament but still reaches only that territory,
      // so it gets the same "only here" treatment as state law — the reader's question is
      // "does this apply to me", not "who enacted it".
      const mismatch = userState && s.state &&
                       userState.toLowerCase() !== s.state.toLowerCase();
      const kind = s.jurisdiction === 'TERRITORY' ? 'Central Act, ' : '';
      badges.push(`<span class="badge juris ${mismatch ? 'mismatch' : 'state'}">`
        + `${kind}${esc(s.state)} only</span>`);
    }

    if (s.status === 'in_force') badges.push('<span class="badge force">in force</span>');
    if (s.status === 'omitted') badges.push('<span class="badge omitted">omitted</span>');
    if (s.status === 'not_in_force') badges.push('<span class="badge omitted">not in force</span>');
    if (s.effective_date) badges.push(`<span class="badge">from ${esc(s.effective_date)}</span>`);
    if (s.category) badges.push(`<span class="badge">${esc(s.category)}</span>`);

    const title = safeUrl(s.url)
      ? `<a href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>`
      : esc(s.title);

    const mismatched = (s.jurisdiction === 'STATE' || s.jurisdiction === 'TERRITORY') &&
                       userState && s.state &&
                       userState.toLowerCase() !== s.state.toLowerCase();
    return `<div class="src${mismatched ? ' mismatch' : ''}" id="src-${esc(s.id)}"
                 data-kind="${esc(s.kind)}">
      <div class="top"><span class="cite" data-cite="${esc(s.id)}">${esc(s.id)}</span>
        <span class="tier">${esc(s.tier_label)}</span></div>
      <div class="title">${title}</div>
      ${s.domain ? `<div class="domain">${esc(s.domain)}</div>` : ''}
      <div class="snippet">${esc(s.snippet)}</div>
      ${(s.caveats || []).map((c) => `<div class="warn-line">${esc(c)}</div>`).join('')}
      ${badges.length ? `<div class="meta">${badges.join('')}</div>` : ''}
      ${mismatched ? `<div class="warn-line">${esc(s.state)} law — it governs matters located
        in ${esc(s.state)}, such as a flat or workplace there, even though you selected
        ${esc(userState)}. It does not govern matters in ${esc(userState)}.</div>` : ''}
    </div>`;
  }).join('');
}

/* The drawer is closed by default and opens on request: the Sources button, the "N sources"
   button under an answer, or a citation chip. */
function setSourcesOpen(open) {
  app.dataset.sources = open ? 'open' : 'closed';
  sourcesBtn.setAttribute('aria-expanded', String(open));
}

function setSourceCount(n) {
  const changed = sourcesCount.textContent !== String(n);
  sourcesCount.textContent = String(n);
  sourcesCount.hidden = !n;
  if (changed && n) {                      // a small pulse says "new sources arrived"
    sourcesCount.classList.remove('bump');
    void sourcesCount.offsetWidth;
    sourcesCount.classList.add('bump');
  }
}

function flashSource(id) {
  setSourcesOpen(true);
  const node = document.getElementById(`src-${id}`);
  if (!node) return;
  node.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  node.classList.add('flash');
  setTimeout(() => node.classList.remove('flash'), 1400);
}

/* ── the stream ────────────────────────────────────────────────────────────────── */
async function ask(question) {
  if (state.busy || state.locked || !question.trim()) return;
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
    finishTurn(await describeRefusal(response));
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

// A refused request carries {error: {kind, message}}. Usage limits get the popup; anything
// else is shown as a notice under the question.
async function describeRefusal(response) {
  let error = null;
  try { error = (await response.json()).error; } catch { /* not JSON */ }
  if (error?.kind === 'auth') {           // signed out, or the session expired
    window.location.href = '/login';
    return error.message;
  }
  if (!error) return `The server returned ${response.status}. Please try again.`;
  if (error.kind === 'client_budget' || error.kind === 'daily_budget') {
    showLimit(error.kind, error.message);
  }
  return error.message;
}

function showQueue(position) {
  if (!turn) return;
  let banner = turn.notices.querySelector('.queue-banner');
  if (position <= 0) { banner?.remove(); return; }
  if (!banner) {
    banner = el('div', 'notice pause queue-banner');
    turn.notices.prepend(banner);
  }
  banner.innerHTML = `<span aria-hidden="true">⏳</span><span class="text">The service is busy `
    + `right now — you are <b>number ${position}</b> in line. Your question will start `
    + `automatically.</span>`;
}

/* The usage dialog: opened by a limit being reached, or from the allowance in the footer.
   With a reset code configured on the server, it also restores the allowance. */
function openUsage(title, message) {
  $('#limitTitle').textContent = title;
  $('#limitText').textContent = message;
  $('#resetForm').hidden = !state.config.budget_reset;
  $('#resetMsg').textContent = '';
  $('#resetMsg').className = 'reset-msg';
  $('#limit').hidden = false;
  ($('#resetForm').hidden ? $('#limitClose') : $('#resetCode')).focus();
}

function showLimit(kind, message) {
  openUsage(kind === 'daily_budget' ? "Today's limit has been reached"
                                    : 'Free usage limit reached', message);
  // The allowance is spent (until it resets); stop inviting more questions.
  state.locked = true;
  input.disabled = true;
  input.placeholder = 'The free usage limit has been reached';
}

function unlock() {
  state.locked = false;
  input.disabled = false;
  input.placeholder = 'Ask about your legal rights…';
}

function describeQuota(q) {
  if (!q?.budget_usd) return 'This service has no per-visitor limit.';
  const reset = q.resets_after_hours ? ` It resets ${q.resets_after_hours} hours after your `
                                       + 'first question.' : '';
  return `You have $${q.remaining_usd.toFixed(2)} of your $${q.budget_usd.toFixed(2)} free `
       + `allowance left. A typical answer costs well under a cent.${reset}`;
}

async function submitReset(event) {
  event.preventDefault();
  const msg = $('#resetMsg');
  const code = $('#resetCode').value.trim();
  if (!code) return;
  try {
    const response = await fetch('/api/quota/reset', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      msg.textContent = body.error?.message || 'That did not work. Please try again.';
      msg.className = 'reset-msg bad';
      return;
    }
    $('#resetCode').value = '';
    msg.textContent = 'Done: your allowance is back to full.';
    msg.className = 'reset-msg good';
    $('#limitText').textContent = describeQuota(body);
    unlock();
    refreshQuota();
  } catch (err) {
    msg.textContent = `Could not reach the server: ${err.message}`;
    msg.className = 'reset-msg bad';
  }
}

// Shown in the footer until the server has finished starting; checked a few times, then left.
async function checkHealth(triesLeft) {
  try {
    const h = await (await fetch('/api/health')).json();
    if (h.status === 'unavailable') { statEl.textContent = 'service unavailable — try again later'; return; }
    if (h.ready) { if (statEl.textContent === 'starting up…') statEl.textContent = ''; return; }
    statEl.textContent = 'starting up…';
  } catch { return; }
  if (triesLeft > 0) setTimeout(() => checkHealth(triesLeft - 1), 3000);
}

async function refreshQuota() {
  try {
    const q = await (await fetch('/api/quota')).json();
    state.quota = q;
    if (q.budget_usd) {
      quotaEl.textContent = `$${q.remaining_usd.toFixed(2)} of $${q.budget_usd.toFixed(2)} left`;
      quotaEl.hidden = false;
    }
    if (q.exhausted && !state.locked) {
      showLimit('client_budget', 'You have used this free service\'s allowance for your '
                + 'connection, so it cannot answer more questions for you.');
    }
  } catch { /* the footer is a courtesy; never fail over it */ }
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'queue': showQueue(ev.position); break;
    case 'limit': showLimit(ev.kind, ev.message); break;

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
      // A rewrite after fact-checking arrives as stage "revise" and is swapped in whole by
      // `answer_revised`. The draft stays readable meanwhile, only visibly marked as being
      // updated — clearing it made the answer vanish mid-read and start over.
      if (ev.id === 'revise' && state.answerEl) {
        state.answerEl.classList.toggle('revising', ev.status === 'running');
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
      if (state.answerEl) {
        state.answerEl.classList.remove('revising');
        state.answerEl.innerHTML = renderMarkdown(state.answerText);
      }
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
        // What this question was billed, from each response's own usage.cost.
        typeof ev.cost_usd === 'number' ? `$${ev.cost_usd.toFixed(4)}` : '',
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
    turn.notices.querySelector('.queue-banner')?.remove();
    if (errorText) addNotice({ level: 'warn', text: errorText });
    if (state.answerEl) {
      // A rewrite that failed or was stopped never sends answer_revised; never leave the draft
      // looking provisional once the turn is over.
      state.answerEl.classList.remove('revising');
      state.answerEl.innerHTML = renderMarkdown(state.answerText);
    }

    const secs = ((Date.now() - turn.started) / 1000).toFixed(1);
    turn.label.textContent = `Research · ${secs}s`;
    turn.timeline.open = false;

    if (state.answerText.trim()) turn.wrap.appendChild(buildVerdict(turn.verdict));
    if (!state.sources.size) renderSources();
  }
  turn = null;
  refreshQuota();
  if (!state.locked) input.focus();
}

function buildVerdict(text) {
  const row = el('div', 'verdict');
  row.innerHTML = `<span class="checked">${esc(text || '')}</span>`;
  const count = state.sources.size;
  if (count) {
    const open = el('button', 'open-sources', `${count} source${count === 1 ? '' : 's'}`);
    open.type = 'button';
    open.onclick = () => setSourcesOpen(true);
    row.appendChild(open);
  }
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
  const suggestion = e.target.closest('.suggestion');
  if (suggestion && !state.busy && !state.locked) {
    input.value = suggestion.querySelector('.s-q').textContent; send();
  }
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
  threadInner.replaceChildren(welcome);
  renderSuggestions();
  state.sources.clear();
  sourcesPane.innerHTML = SOURCES_EMPTY;
  setSourceCount(0);
  setSourcesOpen(false);
  statEl.textContent = '';
  input.focus();
};

// The "How it works" panel shows the server's real budgets, not numbers typed into the page.
function fillDepthFacts(depths) {
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const format = {
    rounds: (d) => (d.rounds > 1 ? `up to ${d.rounds}` : '1'),
    pages: (d) => (!d.pages ? 'none'
      : `up to ${d.pages}${d.link_depth > 1 ? `, ${plural(d.link_depth, 'link')} deep` : ''}`),
    deadline: (d) => `${Math.round(d.deadline_s)} s`,
  };
  document.querySelectorAll('[data-fact]').forEach((node) => {
    const [depth, fact] = node.dataset.fact.split('.');
    if (depths[depth] && format[fact]) node.textContent = format[fact](depths[depth]);
  });
}

/* ── boot ──────────────────────────────────────────────────────────────────────── */
(function boot() {
  const saved = localStorage.getItem('kyr.theme');
  if (saved) document.documentElement.dataset.theme = saved;
  else if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    document.documentElement.dataset.theme = 'dark';
  }

  fetch('/api/config').then((r) => r.json()).then((c) => {
    state.config = c;
    fillDepthFacts(c.depths || {});
    const select = $('#state');
    for (const name of c.states || []) {
      const option = el('option');
      option.value = name; option.textContent = name;
      select.appendChild(option);
    }
    // A saved choice that is no longer a valid state falls back to All India.
    select.value = (c.states || []).includes(state.userState) ? state.userState : '';
    state.userState = select.value;
    if (c.disclaimer) $('#disclaimer').textContent = c.disclaimer;
    if (c.user) {
      const signOut = $('#signOut');
      signOut.hidden = false;
      signOut.querySelector('button').title = `Signed in as ${c.user}. Sign out`;
    }
  }).catch(() => {});
  renderSuggestions();
  $('#shuffle').addEventListener('click', () => renderSuggestions());
  checkHealth(10);
  refreshQuota();

  // ── dialogs and the sources drawer ───────────────────────────────────────────────
  const closeUsage = () => { $('#limit').hidden = true; input.focus(); };
  $('#limitClose').addEventListener('click', closeUsage);
  $('#limit').addEventListener('click', (e) => { if (e.target === $('#limit')) closeUsage(); });
  $('#resetForm').addEventListener('submit', submitReset);
  quotaEl.addEventListener('click', () => openUsage('Your allowance', describeQuota(state.quota)));

  sourcesBtn.addEventListener('click', () => setSourcesOpen(app.dataset.sources !== 'open'));
  $('#sourcesClose').addEventListener('click', () => setSourcesOpen(false));
  $('#scrim').addEventListener('click', () => setSourcesOpen(false));
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!$('#limit').hidden) closeUsage();
    else if (app.dataset.sources === 'open') setSourcesOpen(false);
  });

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
