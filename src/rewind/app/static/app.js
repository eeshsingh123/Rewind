"use strict";

const POLL_MS = 600;
const MIN_BOX = 0.01;

const $ = (id) => document.getElementById(id);

const state = {
  id: null,
  session: null,
  media: null,
  shown: 0, // exchanges already in the log
  working: false,
  marked: null, // {box, t}
  marking: false,
};

// Model text only ever goes in as text nodes.
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function clock(seconds) {
  const whole = Math.floor(seconds);
  const minutes = String(Math.floor(whole / 60)).padStart(2, "0");
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function problemOf(response) {
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") return detail;
  } catch {
    // Not JSON: the plain line below is all there is to say.
  }
  return "Something went wrong. Please try again.";
}

function show(which) {
  $("home").hidden = which !== "home";
  $("desk").hidden = which !== "desk";
  $("copy-link").hidden = which !== "desk";
}

// Home

function openHome() {
  show("home");
  loadSamples();
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  if (!response.ok) return;
  const samples = await response.json();
  $("samples").hidden = samples.length === 0;
  $("sample-list").replaceChildren(
    ...samples.map((sample) =>
      el(
        "button",
        { className: "sample", type: "button", onclick: () => openSample(sample.name) },
        el("img", { src: sample.picture_url, alt: "" }),
        el("span", { textContent: sample.title }),
      ),
    ),
  );
}

async function openSample(name) {
  $("home-problem").hidden = true;
  const response = await fetch(`/api/samples/${encodeURIComponent(name)}`, { method: "POST" });
  if (!response.ok) return homeProblem(await problemOf(response));
  go((await response.json()).id);
}

function homeProblem(text) {
  $("home-problem").textContent = text;
  $("home-problem").hidden = false;
  $("drop-text").textContent = "Drop a photo or a video here, or tap to choose";
  $("upload-bar").hidden = true;
}

// XMLHttpRequest, since fetch cannot report upload progress.
function upload(file) {
  $("home-problem").hidden = true;
  $("drop-text").textContent = `Sending ${file.name}`;
  $("upload-bar").hidden = false;
  const fill = $("upload-bar").firstElementChild;
  fill.style.width = "0";
  const form = new FormData();
  form.append("file", file);
  const request = new XMLHttpRequest();
  request.open("POST", "/api/sessions");
  request.responseType = "json";
  request.upload.onprogress = (event) => {
    if (event.lengthComputable) fill.style.width = `${(100 * event.loaded) / event.total}%`;
  };
  request.onload = () => {
    if (request.status === 201) return go(request.response.id);
    homeProblem(request.response?.detail || "That upload did not work. Please try again.");
  };
  request.onerror = () => homeProblem("That upload did not work. Please try again.");
  request.send(form);
}

function go(id) {
  history.pushState({}, "", `/s/${id}`);
  openSession(id);
}

// A session

async function openSession(id) {
  state.id = id;
  show("desk");
  await refresh();
  await follow();
}

async function refresh() {
  const response = await fetch(`/api/sessions/${state.id}`);
  if (!response.ok) return;
  const session = await response.json();
  const first = !state.session;
  state.session = session;
  if (first) showFootage(session);
  session.exchanges.slice(state.shown).forEach(showExchange);
  state.shown = session.exchanges.length;
  showSuggested();
  $("log").scrollTop = $("log").scrollHeight;
}

// Follows whatever the session is doing until it stops. Polling works through any tunnel or proxy.
async function follow() {
  setWorking(true);
  let turn = null;
  for (;;) {
    const response = await fetch(`/api/sessions/${state.id}/live`);
    if (response.status === 404) return notFound();
    if (!response.ok) {
      await sleep(POLL_MS);
      continue;
    }
    const live = await response.json();
    if (live.question && !state.session) await refresh();
    if (!live.working) {
      turn?.remove();
      await refresh();
      if (live.problem) showProblem(live);
      setWorking(false);
      return;
    }
    turn = showLive(turn, live);
    await sleep(POLL_MS);
  }
}

function notFound() {
  setWorking(false);
  $("waiting").textContent = "There is nothing at this link.";
  $("log").replaceChildren(el("p", { textContent: "Start again from the home page." }));
}

function setWorking(working) {
  state.working = working;
  $("send").disabled = working;
  for (const button of $("suggested").querySelectorAll("button")) button.disabled = working;
}

function showLive(turn, live) {
  const steps = el("ul", { className: "steps" });
  live.steps.forEach((step, index) => {
    const last = index === live.steps.length - 1;
    // A step still running is replaced by the next one.
    if (!step.done && !last) return;
    const status = step.done ? "done" : "now";
    steps.append(el("li", { className: status, textContent: step.text }));
  });
  const next = el(
    "div",
    { className: "turn live" },
    live.question ? el("p", { className: "asked", textContent: live.question }) : null,
    steps,
  );
  if (turn) turn.replaceWith(next);
  else $("log").append(next);
  $("log").scrollTop = $("log").scrollHeight;
  return next;
}

function showProblem(live) {
  if (!state.session) $("waiting").textContent = live.problem;
  $("log").append(
    el(
      "div",
      { className: "turn" },
      live.question ? el("p", { className: "asked", textContent: live.question }) : null,
      el("p", { className: "problem", textContent: live.problem }),
    ),
  );
  $("log").scrollTop = $("log").scrollHeight;
}

function showFootage(session) {
  const frame = $("frame");
  $("waiting").hidden = true;
  const media = session.photo
    ? el("img", { src: session.media_url, alt: "The uploaded photo" })
    : el("video", {
        src: session.media_url,
        controls: true,
        playsInline: true,
        preload: "metadata",
      });
  state.media = media;
  frame.prepend(media);
  $("mark").hidden = false;
  $("mark").textContent = session.photo ? "Mark a part" : "Mark something";
  if (!session.photo) {
    showTimeline(session);
    media.addEventListener("timeupdate", onTime);
    media.addEventListener("seeking", onTime);
  }
  $("log").append(firstLook(session));
}

function firstLook(session) {
  const notes = el("ul", { className: "notes" });
  for (const moment of session.moments) {
    notes.append(
      el("li", {}, session.photo ? null : timeButton(moment.start_s), session.photo ? null : " ", moment.what),
    );
  }
  return el(
    "div",
    { className: "first" },
    el("p", { textContent: session.summary }),
    session.moments.length ? notes : null,
  );
}

function timeButton(seconds) {
  return el("button", {
    className: "time",
    type: "button",
    textContent: clock(seconds),
    onclick: () => seek(seconds),
  });
}

function seek(seconds) {
  const video = state.media;
  if (!video || state.session?.photo) return;
  video.currentTime = seconds;
  video.play().catch(() => {});
  if (matchMedia("(max-width: 860px)").matches) {
    video.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function showTimeline(session) {
  $("timeline").hidden = false;
  $("end-time").textContent = clock(session.duration_s);
  for (const moment of session.moments) addTick(moment.start_s, moment.end_s, moment.what, "look");
  $("track").addEventListener("click", (event) => {
    if (event.target.classList.contains("tick")) return;
    const { left, width } = $("track").getBoundingClientRect();
    seek(((event.clientX - left) / width) * session.duration_s);
  });
}

function addTick(start, end, text, kind) {
  const duration = state.session.duration_s;
  const tick = el("button", {
    className: `tick ${kind}`,
    type: "button",
    title: `${clock(start)} ${text}`,
    onclick: () => seek(start),
  });
  tick.setAttribute("aria-label", `${clock(start)}: ${text}`);
  tick.style.left = `${(100 * start) / duration}%`;
  tick.style.width = `${(100 * (end - start)) / duration}%`;
  $("track").append(tick);
}

function onTime() {
  const video = state.media;
  if (video.duration) $("playhead").style.left = `${(100 * video.currentTime) / video.duration}%`;
  showBox();
}

function showExchange(exchange) {
  const turn = el(
    "div",
    { className: "turn" },
    el("p", { className: "asked" }, exchange.question, marktag(exchange)),
  );
  if (exchange.steps.length) {
    const steps = el("ul", { className: "steps" });
    for (const step of exchange.steps) {
      steps.append(el("li", { className: "done", textContent: step }));
    }
    const count = exchange.steps.length;
    turn.append(
      el("details", {}, el("summary", { textContent: `${count} step${count > 1 ? "s" : ""}` }), steps),
    );
  }
  if (exchange.findings.length) {
    exchange.findings.forEach((finding) => turn.append(showFinding(finding)));
  } else {
    turn.append(el("p", { textContent: exchange.answer }));
  }
  turn.append(el("p", { className: "cost", textContent: costLine(exchange.cost_usd) }));
  $("log").append(turn);
}

function marktag(exchange) {
  if (exchange.marked_at_s === null) return null;
  const where = state.session?.photo ? "marked part" : `marked at ${clock(exchange.marked_at_s)}`;
  return el("small", { textContent: `  ${where}` });
}

function showFinding(finding) {
  const meta = el("div", { className: "meta" });
  if (finding.start_s !== null) {
    meta.append(timeButton(finding.start_s));
    addTick(finding.start_s, finding.end_s, finding.text, "found");
  }
  meta.append(finding.kind === "measured" ? "Measured by code" : "Seen by Muse");
  return el(
    "div",
    { className: "finding" },
    el("p", { textContent: finding.text }),
    meta,
    finding.proof_url ? proof(finding) : null,
  );
}

function proof(finding) {
  if (finding.proof_photo) {
    return el(
      "a",
      { href: finding.proof_url, target: "_blank", rel: "noopener" },
      el("img", { className: "proof", src: finding.proof_url, alt: "Proof: the photo with boxes drawn" }),
    );
  }
  return el("video", {
    className: "proof",
    src: finding.proof_url,
    controls: true,
    muted: true,
    playsInline: true,
    preload: "metadata",
  });
}

function costLine(cost) {
  const usd = Number(cost);
  if (usd === 0) return "No cost: already answered";
  return usd < 0.001 ? "cost under $0.001" : `cost $${usd.toFixed(3)}`;
}

function showSuggested() {
  const asked = new Set(state.session.exchanges.map((exchange) => exchange.question));
  const left = state.session.questions.filter((question) => !asked.has(question));
  $("suggested").replaceChildren(
    ...left.map((question) =>
      el("button", {
        type: "button",
        textContent: question,
        disabled: state.working,
        onclick: () => ask(question),
      }),
    ),
  );
}

// Asking

async function ask(question) {
  if (state.working || !state.session) return;
  const marked = state.marked;
  const body = { question };
  if (marked) Object.assign(body, { box: marked.box, at_s: marked.t });
  setWorking(true);
  const response = await fetch(`/api/sessions/${state.id}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    showProblem({ question, problem: await problemOf(response) });
    setWorking(false);
    return;
  }
  $("question").value = "";
  clearMark();
  await follow();
}

function onSubmit(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (question) return ask(question);
  if (state.marked && !state.session.photo) ask("Follow this");
}

// Marking a box on the frame

function startMarking() {
  if (state.marking) return stopMarking();
  state.marking = true;
  if (!state.session.photo) state.media.pause();
  $("mark").classList.add("on");
  $("marker").hidden = false;
  $("marker").classList.remove("resting");
  $("mark-hint").hidden = false;
  $("mark-hint").textContent = state.session.photo
    ? "Drag a box around the part you mean"
    : "Pause on the right moment, then drag a box around it";
}

function stopMarking() {
  state.marking = false;
  $("mark").classList.remove("on");
  $("mark-hint").hidden = true;
  $("marker").classList.add("resting");
  $("marker").hidden = !state.marked;
}

// Where the picture sits inside its element: it is letterboxed when the frame is capped in height.
function pictureRect() {
  const media = state.media;
  const width = media.videoWidth || media.naturalWidth;
  const height = media.videoHeight || media.naturalHeight;
  const scale = Math.min(media.clientWidth / width, media.clientHeight / height);
  const shown = { width: width * scale, height: height * scale };
  return {
    ...shown,
    left: (media.clientWidth - shown.width) / 2,
    top: (media.clientHeight - shown.height) / 2,
  };
}

function fraction(event, rect) {
  const bounds = $("marker").getBoundingClientRect();
  const clamp = (value) => Math.min(1, Math.max(0, value));
  return {
    x: clamp((event.clientX - bounds.left - rect.left) / rect.width),
    y: clamp((event.clientY - bounds.top - rect.top) / rect.height),
  };
}

function drawBox(box) {
  const rect = pictureRect();
  const style = $("box").style;
  style.left = `${rect.left + box.x0 * rect.width}px`;
  style.top = `${rect.top + box.y0 * rect.height}px`;
  style.width = `${(box.x1 - box.x0) * rect.width}px`;
  style.height = `${(box.y1 - box.y0) * rect.height}px`;
  $("box").hidden = false;
}

function showBox() {
  const { marked, media, session } = state;
  if (!marked) return;
  const here = session.photo || Math.abs(media.currentTime - marked.t) < 0.3;
  $("box").hidden = !here;
}

function onDragStart(event) {
  const rect = pictureRect();
  const start = fraction(event, rect);
  const boxFrom = (point) => ({
    x0: Math.min(start.x, point.x),
    y0: Math.min(start.y, point.y),
    x1: Math.max(start.x, point.x),
    y1: Math.max(start.y, point.y),
  });
  const marker = $("marker");
  marker.setPointerCapture(event.pointerId);
  const onMove = (move) => drawBox(boxFrom(fraction(move, rect)));
  const onUp = (up) => {
    marker.removeEventListener("pointermove", onMove);
    marker.removeEventListener("pointerup", onUp);
    const box = boxFrom(fraction(up, rect));
    if (box.x1 - box.x0 < MIN_BOX || box.y1 - box.y0 < MIN_BOX) {
      if (!state.marked) $("box").hidden = true;
      return;
    }
    setMark(box);
  };
  marker.addEventListener("pointermove", onMove);
  marker.addEventListener("pointerup", onUp);
}

function setMark(box) {
  const photo = state.session.photo;
  state.marked = { box, t: photo ? 0 : state.media.currentTime };
  stopMarking();
  $("marked-text").textContent = photo ? "Part of the photo marked" : `Marked at ${clock(state.marked.t)}`;
  $("marked-chip").hidden = false;
  $("question").placeholder = photo ? "What do you want to know about this part?" : "Follow this, or ask about it";
  $("question").focus();
}

function clearMark() {
  state.marked = null;
  $("marked-chip").hidden = true;
  $("box").hidden = true;
  $("marker").hidden = true;
  $("question").placeholder = "Ask anything";
}

// Wiring

function wire() {
  $("file").addEventListener("change", () => {
    if ($("file").files[0]) upload($("file").files[0]);
  });
  const drop = $("drop");
  for (const name of ["dragenter", "dragover"]) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("over");
    });
  }
  drop.addEventListener("drop", (event) => {
    if (event.dataTransfer.files[0]) upload(event.dataTransfer.files[0]);
  });

  $("ask").addEventListener("submit", onSubmit);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("ask").requestSubmit();
    }
  });
  $("mark").addEventListener("click", startMarking);
  $("marker").addEventListener("pointerdown", onDragStart);
  $("unmark").addEventListener("click", clearMark);
  $("copy-link").addEventListener("click", async () => {
    await navigator.clipboard.writeText(location.href);
    $("copy-link").textContent = "Link copied";
    await sleep(1500);
    $("copy-link").textContent = "Copy link";
  });
  addEventListener("popstate", () => location.reload());
}

wire();
const opened = location.pathname.match(/^\/s\/([^/]+)/);
if (opened) openSession(opened[1]);
else openHome();
