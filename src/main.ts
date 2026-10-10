import { AgentLoop, shouldHold, type DecisionRecord } from "./agent/loop.ts";
import { DecisionApiError } from "./agent/client.ts";
import { randomByName, systemOnePolicy } from "./agent/policies.ts";
import { llmPolicy } from "./agent/llm.ts";
import { decisionsPolicy, DECISIONS_DEFAULT_BASE_URL } from "./agent/decisions.ts";
import { nativeBridge, nativeChatPolicy, nativeJevPolicy, nativePolicy } from "./agent/native.ts";
import { DEFAULT_PLAYER, isPlayerId, playerLabel, type PlayerId } from "./agent/players.ts";
import { GameAudio } from "./audio/player.ts";
import { findDecisionPoint } from "./engine/decision.ts";
import { createGame, step, TPS } from "./engine/game.ts";
import { opposite, type Dir, type GameState, type StepInput } from "./engine/types.ts";
import { DEFAULT_ENCODER, ENCODERS } from "./encoders/index.ts";
import { HEIGHT, render, WIDTH, type Overlay } from "./render/canvas.ts";

const params = new URLSearchParams(location.search);
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- configuration -----------------------------------------------------------
// Inside the iOS app, decisions go through native bridges instead of an HTTP endpoint, and the model is
// one of the app's players (see src/agent/players.ts), chosen by the app.
const bridge = nativeBridge();
const defaultEndpoint = import.meta.env.DEV ? "/decide" : (import.meta.env.VITE_DECISION_BASE_URL ?? "http://localhost:11434");
const initialPlayer = (): PlayerId => {
  const asked = params.get("model");
  return isPlayerId(asked) ? asked : DEFAULT_PLAYER;
};
const config = {
  mode: (params.get("mode") ?? "ai") as "ai" | "human",
  model: bridge ? initialPlayer() : (params.get("model") ?? import.meta.env.VITE_DEFAULT_MODEL ?? "tev1:4b"),
  encoder: params.get("encoder") ?? DEFAULT_ENCODER,
  speed: Number(params.get("speed") ?? "1"),
  /** Hold the game at a junction until the model answers (for slow on-device models). */
  waitForModel: params.get("wait") === "1",
  endpoint: params.get("endpoint") ?? defaultEndpoint,
  seed: Number(params.get("seed") ?? Math.floor(Math.random() * 1e6)),
};

// ---- canvas ------------------------------------------------------------------
const canvas = $<HTMLCanvasElement>("game");
const dpr = Math.min(2, window.devicePixelRatio || 1);
canvas.width = WIDTH * dpr;
canvas.height = HEIGHT * dpr;
// Scale down to fit narrow screens while keeping the aspect ratio.
canvas.style.width = "100%";
canvas.style.maxWidth = `${WIDTH}px`;
canvas.style.height = "auto";
const ctx = canvas.getContext("2d")!;

// ---- game state ----------------------------------------------------------------
let game: GameState = createGame(config.seed, { speed: config.speed });
let pending: StepInput = {};
let paused = false;
const records: DecisionRecord[] = [];
let lastRecord: DecisionRecord | null = null;
const overlay: Overlay = {};

// ---- audio -------------------------------------------------------------------
function loadMuted(): boolean {
  try {
    return localStorage.getItem("pacman.muted") === "1";
  } catch {
    return false;
  }
}

const audio = new GameAudio(params.get("sound") === "0" || loadMuted());

function setMuted(muted: boolean): void {
  audio.setMuted(muted);
  $("sound").textContent = muted ? "Sound: off" : "Sound: on";
  try {
    localStorage.setItem("pacman.muted", muted ? "1" : "0");
  } catch {
    // Remembering the choice is a convenience.
  }
}

// Browsers start audio only after a user gesture; the iOS app allows it right away.
for (const type of ["pointerdown", "keydown", "touchstart"]) window.addEventListener(type, () => audio.unlock(), { passive: true });
if (bridge) audio.unlock();
document.addEventListener("visibilitychange", () => audio.setSuspended(document.hidden || paused));

function queue(input: StepInput): void {
  if (input.intent !== undefined) pending.intent = input.intent;
  if (input.reverse) pending.reverse = true;
  if (input.turn) pending.turn = input.turn;
}

function restart(): void {
  game = createGame(Math.floor(Math.random() * 1e6), { speed: config.speed });
  pending = {};
  lastRecord = null;
}

// ---- agent -------------------------------------------------------------------
function appPolicy(player: PlayerId) {
  if (player === "phi4-mini") return nativeChatPolicy(bridge!, playerLabel(player), player);
  if (player === "jev") {
    const jev = nativeJevPolicy({ model: "jev-latest" });
    if (jev) return jev;
  }
  return nativePolicy(bridge!, playerLabel("finetuned"), "finetuned");
}

/** The name shown in the HUD and the status heartbeat. */
function modelName(): string {
  return bridge && isPlayerId(config.model) ? playerLabel(config.model) : config.model;
}

function makePolicy() {
  if (bridge) return appPolicy(isPlayerId(config.model) ? config.model : DEFAULT_PLAYER);
  // "random" and "random-1s" are scripted baselines that need no model.
  const random = randomByName(config.model, config.seed);
  if (random) return random;
  // "jev-latest" (TypeSafe's hosted Jev) goes through the dev server's /jev proxy, which adds TYPESAFE_API_KEY.
  if (import.meta.env.DEV && config.model.startsWith("jev-")) return systemOnePolicy({ baseUrl: "/jev", model: config.model, hosted: true });
  // "llm:<model>" is a plain Ollama chat model answering in constrained JSON, not a decision model.
  if (config.model.startsWith("llm:")) return llmPolicy({ baseUrl: config.endpoint, model: config.model.slice("llm:".length) });
  // "openai:<model>" is OpenAI's Decisions API. The dev server proxies /openai and adds OPENAI_API_KEY, so the key never reaches the page.
  if (config.model.startsWith("openai:")) {
    return decisionsPolicy({ baseUrl: import.meta.env.DEV ? "/openai" : DECISIONS_DEFAULT_BASE_URL, model: config.model.slice("openai:".length) });
  }
  return systemOnePolicy({ baseUrl: config.endpoint, model: config.model, hosted: import.meta.env.VITE_DECISION_HOSTED === "1" });
}

const agent = new AgentLoop(
  {
    getState: () => game,
    apply: queue,
    onDecision: (rec) => {
      records.push(rec);
      if (records.length > 5000) records.shift();
      lastRecord = rec;
      hideError();
    },
    onError: (err) => showError(err),
  },
  makePolicy(),
  ENCODERS[config.encoder] ?? ENCODERS[DEFAULT_ENCODER],
);
agent.waitForModel = config.waitForModel;

function syncAgent(): void {
  agent.policy = makePolicy();
  agent.encoder = ENCODERS[config.encoder] ?? ENCODERS[DEFAULT_ENCODER];
  if (config.mode === "ai" && !paused) agent.start();
  else agent.stop();
}

// ---- fixed-step loop -------------------------------------------------------------
const STEP_MS = 1000 / TPS;
let acc = 0;
let last = performance.now();
const tickTimes: number[] = [];

// Frame timing, reported in the status heartbeat to diagnose stalls.
const frameStats = { frames: 0, ticks: 0, maxGapMs: 0, droppedMs: 0 };

function frame(now: number): void {
  const gap = now - last;
  frameStats.frames++;
  frameStats.maxGapMs = Math.max(frameStats.maxGapMs, gap);
  if (gap > 250) frameStats.droppedMs += gap - 250;
  const dt = Math.min(250, gap);
  last = now;
  if (!paused) {
    acc += dt;
    let steps = 0;
    while (acc >= STEP_MS && steps < 5) {
      if (config.mode === "ai" && config.waitForModel && shouldHold(game, agent.pending)) {
        acc = 0;
        break;
      }
      step(game, pending);
      audio.onTick(game);
      pending = {};
      acc -= STEP_MS;
      steps++;
      tickTimes.push(now);
      frameStats.ticks++;
    }
    if (steps === 5) {
      frameStats.droppedMs += acc;
      acc = 0;
    }
  }
  while (tickTimes.length && now - tickTimes[0] > 1000) tickTimes.shift();

  if (config.mode === "ai" && game.phase === "playing") {
    const dp = findDecisionPoint(game);
    overlay.junction = { x: dp.x, y: dp.y };
    overlay.arrow =
      lastRecord && !lastRecord.stale && lastRecord.junction === dp.key && lastRecord.choice !== "back"
        ? (lastRecord.choice as Dir)
        : null;
  } else {
    overlay.junction = undefined;
    overlay.arrow = null;
  }
  render(ctx, game, overlay);
  requestAnimationFrame(frame);
}

// ---- HUD ---------------------------------------------------------------------
function pct(xs: number[], p: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

function updateHud(): void {
  $("score").textContent = String(game.score);
  $("lives").textContent = String(game.lives);
  $("level").textContent = String(game.level);
  $("pellets").textContent = String(game.foodLeft);
  $("tps").textContent = String(tickTimes.length);
  const holding = config.mode === "ai" && config.waitForModel && shouldHold(game, agent.pending);
  $("who").textContent = config.mode === "ai" ? `${modelName()} · ${config.encoder}${holding ? " · waiting for the model" : ""}` : "keyboard";

  const recent = records.slice(-100);
  const lat = recent.map((r) => r.latencyMs);
  const nowMs = Date.now();
  const lastFive = records.filter((r) => nowMs - r.time < 5000).length;
  $("lat-last").textContent = lastRecord ? `${Math.round(lastRecord.latencyMs)} ms` : "-";
  $("lat-p50").textContent = lat.length ? `${Math.round(pct(lat, 0.5))} ms` : "-";
  $("lat-p90").textContent = lat.length ? `${Math.round(pct(lat, 0.9))} ms` : "-";
  $("dps").textContent = config.mode === "ai" ? (lastFive / 5).toFixed(1) : "-";
  $("stale").textContent = recent.length ? `${Math.round((100 * recent.filter((r) => r.stale).length) / recent.length)}%` : "-";

  const bars = $("bars");
  bars.innerHTML = "";
  if (lastRecord?.probabilities) {
    $("confidence").textContent = `confidence ${lastRecord.confidence?.toFixed(2)}`;
    for (const key of lastRecord.keys) {
      const p = lastRecord.probabilities[key] ?? 0;
      const row = document.createElement("div");
      row.className = `bar${key === lastRecord.choice ? " chosen" : ""}`;
      row.innerHTML = `<span class="label">${key}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span>${(p * 100).toFixed(0)}%</span>`;
      bars.appendChild(row);
    }
    $("state-view").textContent = JSON.stringify(lastRecord.state, null, 1);
  } else {
    $("confidence").textContent = "";
    $("state-view").textContent = config.mode === "ai" ? "waiting for the first decision..." : "-";
  }
}

// ---- errors ------------------------------------------------------------------
function showError(err: unknown): void {
  const box = $("error");
  box.classList.add("show");
  if (err instanceof DecisionApiError) {
    $("error-message").textContent = err.message;
    $("error-hint").textContent = err.hint;
  } else {
    $("error-message").textContent = String(err);
    $("error-hint").textContent = "";
  }
}

function hideError(): void {
  $("error").classList.remove("show");
}

// ---- controls ----------------------------------------------------------------
const KEYS: Record<string, Dir> = {
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
  w: "up", s: "down", a: "left", d: "right",
};

window.addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).tagName === "INPUT") return;
  if (e.key === " ") {
    togglePause();
    e.preventDefault();
    return;
  }
  if (e.key === "r" || e.key === "R") {
    restart();
    return;
  }
  if (e.key === "m" || e.key === "M") {
    setMuted(!audio.muted);
    return;
  }
  const d = KEYS[e.key];
  if (!d || config.mode !== "human") return;
  e.preventDefault();
  if (game.pac.dir && d === opposite(game.pac.dir)) queue({ reverse: true });
  else queue({ intent: d });
});

// Swipes steer Pac-Man in keyboard mode on touch screens.
let touchStart: { x: number; y: number } | null = null;
canvas.addEventListener("touchstart", (e) => {
  const t = e.changedTouches[0];
  touchStart = { x: t.clientX, y: t.clientY };
}, { passive: true });
canvas.addEventListener("touchend", (e) => {
  if (!touchStart || config.mode !== "human") return;
  const t = e.changedTouches[0];
  const dx = t.clientX - touchStart.x;
  const dy = t.clientY - touchStart.y;
  touchStart = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 20) return;
  const d: Dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : dy > 0 ? "down" : "up";
  if (game.pac.dir && d === opposite(game.pac.dir)) queue({ reverse: true });
  else queue({ intent: d });
}, { passive: true });

function setWaitForModel(value: boolean): void {
  config.waitForModel = value;
  agent.waitForModel = value;
  $<HTMLInputElement>("wait").checked = value;
}

function togglePause(): void {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  audio.setSuspended(paused);
  syncAgent();
}

function setupControls(): void {
  const mode = $<HTMLSelectElement>("mode");
  const model = $<HTMLInputElement>("model");
  const encoder = $<HTMLSelectElement>("encoder");
  const speed = $<HTMLSelectElement>("speed");
  const endpoint = $<HTMLInputElement>("endpoint");

  for (const name of Object.keys(ENCODERS)) encoder.add(new Option(name, name));
  if (bridge) {
    // The app picks the player natively; hide the HTTP endpoint and lock the model field.
    model.disabled = true;
    for (const el of [endpoint, endpoint.previousElementSibling]) (el as HTMLElement).style.display = "none";
  }
  mode.value = config.mode;
  model.value = config.model;
  encoder.value = config.encoder;
  speed.value = String(config.speed);
  endpoint.value = config.endpoint;

  mode.onchange = () => {
    config.mode = mode.value as "ai" | "human";
    syncAgent();
  };
  model.onchange = () => setModel(model.value.trim());
  encoder.onchange = () => {
    config.encoder = encoder.value;
    syncAgent();
  };
  const wait = $<HTMLInputElement>("wait");
  wait.checked = config.waitForModel;
  wait.onchange = () => setWaitForModel(wait.checked);
  speed.onchange = () => {
    config.speed = Number(speed.value);
    game.options.speed = config.speed;
  };
  endpoint.onchange = () => {
    config.endpoint = endpoint.value.trim();
    loadModels();
    syncAgent();
  };
  $("pause").onclick = togglePause;
  $("restart").onclick = restart;
  $("sound").onclick = () => setMuted(!audio.muted);
  setMuted(audio.muted);
  $("export").onclick = () => {
    const blob = new Blob(records.map((r) => JSON.stringify(r) + "\n"), { type: "application/x-ndjson" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `decisions-${Date.now()}.jsonl`;
    a.click();
  };
}

/** Players offered in the model list whatever the endpoint serves: hosted Jev (dev server only) and the scripted ones. */
const BUILT_IN_PLAYERS = [...(import.meta.env.DEV ? ["jev-latest"] : []), "random", "random-1s", "random-2s"];

async function loadModels(): Promise<void> {
  if (bridge) return;
  const list = $("model-list");
  list.innerHTML = "";
  for (const name of BUILT_IN_PLAYERS) list.appendChild(new Option(name, name));
  try {
    const res = await fetch(`${config.endpoint.replace(/\/$/, "")}/api/tags`);
    const body = (await res.json()) as { models?: { name: string }[] };
    for (const m of body.models ?? []) list.appendChild(new Option(m.name, m.name));
  } catch {
    // The model list is a convenience; decisions report their own errors.
  }
}

/** Switches the model (in the app, the player); latency and the probability bars start over for it. */
function setModel(value: string): void {
  config.model = value;
  $<HTMLInputElement>("model").value = value;
  records.length = 0;
  lastRecord = null;
  hideError();
  syncAgent();
}

// Inside the iOS app, report game status so the run can be checked from the Mac.
const statusBridge = (window as unknown as { webkit?: { messageHandlers?: { status?: { postMessage(m: unknown): void } } } })
  .webkit?.messageHandlers?.status;
if (statusBridge) {
  let seq = 0;
  const median = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null);
  setInterval(() => {
    const recent = records.slice(-50);
    statusBridge.postMessage({
      seq: ++seq, tick: game.tick, phase: game.phase, score: game.score, lives: game.lives, level: game.level,
      pelletsLeft: game.foodLeft, decisions: records.length, stale: records.filter((r) => r.stale).length,
      latencyP50: median(recent.map((r) => r.latencyMs)),
      tokensP50: median(recent.flatMap((r) => (r.inputTokens ? [r.inputTokens] : []))),
      ticksPerSecond: tickTimes.length, mode: config.mode, model: config.model, modelName: modelName(), paused, waitForModel: config.waitForModel,
      window: { seconds: 2, ...frameStats, maxGapMs: Math.round(frameStats.maxGapMs), droppedMs: Math.round(frameStats.droppedMs) },
      audio: audio.state, muted: audio.muted,
    });
    Object.assign(frameStats, { frames: 0, ticks: 0, maxGapMs: 0, droppedMs: 0 });
  }, 2000);
}

setupControls();
void loadModels();
syncAgent();
setInterval(updateHud, 100);
requestAnimationFrame(frame);

// Exposed for scripted recording, debugging, and native control (the iOS app calls these).
(window as unknown as { __pacman: object }).__pacman = {
  get game() {
    return game;
  },
  records,
  setMode(mode: "ai" | "human") {
    config.mode = mode;
    $<HTMLSelectElement>("mode").value = mode;
    syncAgent();
  },
  setPaused(value: boolean) {
    if (paused !== value) togglePause();
  },
  setWaitForModel(value: boolean) {
    setWaitForModel(value);
  },
  setSpeed(value: number) {
    config.speed = value;
    game.options.speed = value;
    $<HTMLSelectElement>("speed").value = String(value);
  },
  setModel(value: string) {
    if (bridge && !isPlayerId(value)) return;
    if (value !== config.model) setModel(value);
  },
  restart,
  audio,
};
