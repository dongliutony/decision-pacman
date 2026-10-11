import { findDecisionPoint, isStillAhead, pacTilesPerSecond, type DecisionPoint } from "../engine/decision.ts";
import { TPS } from "../engine/game.ts";
import type { GameState, StepInput } from "../engine/types.ts";
import { optionToInput, withoutBack, type Encoder } from "../encoders/types.ts";
import type { Policy } from "./policies.ts";

export interface DecisionRecord {
  time: number;
  model: string;
  encoder: string;
  junction: string;
  keys: string[];
  choice: string;
  probabilities?: Record<string, number>;
  confidence?: number;
  latencyMs: number;
  inputTokens?: number;
  askedTick: number;
  answeredTick: number;
  /** True when the answer arrived after Pac-Man had left that junction. */
  stale: boolean;
  state: unknown;
}

export interface AgentHooks {
  getState(): GameState;
  /** Queue input for the next game tick. */
  apply(input: StepInput): void;
  onDecision(rec: DecisionRecord): void;
  onError(err: unknown): void;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * How far ahead (in tiles) to plan: the distance Pac-Man covers while an
 * answer is in flight, with some margin.
 */
export function lookaheadTiles(s: GameState, expectedLatencyMs: number): number {
  return pacTilesPerSecond(s) * (expectedLatencyMs / 1000) * 1.25 + 0.3;
}

/**
 * In wait-for-model mode, whether the game should hold this tick: a question about `pending` is in
 * flight and Pac-Man would reach that junction's center within the next tick. Ghosts and timers hold
 * too, so a slow model is judged on its choices, not its speed.
 */
export function shouldHold(s: GameState, pending: DecisionPoint | null): boolean {
  if (!pending || s.phase !== "playing") return false;
  const dp = findDecisionPoint(s);
  return dp.key === pending.key && dp.distance <= pacTilesPerSecond(s) / TPS + 0.05;
}

/** Exponential moving average of decision latency, used to size the lookahead. */
export class LatencyEstimate {
  constructor(public value = 250) {}
  update(ms: number): void {
    if (ms > 5) this.value = 0.8 * this.value + 0.2 * ms;
  }
}

/**
 * Blocks "back" for `ticks` after Pac-Man's next junction changes, which happens when Pac-Man turns
 * around or passes a junction (corners do not count). Without it, a model with no memory can answer
 * "back", then "back" again a moment later, and Pac-Man dithers. 0 turns it off.
 */
export class ReverseLock {
  private key: string | null = null;
  private since = -Infinity;
  constructor(public ticks = 0) {}

  /** Call at every question with the game and its next junction (no lookahead); true while "back" is blocked. */
  blocked(s: GameState, junctionKey: string): boolean {
    if (s.tick < this.since) this.since = -Infinity; // a new game started
    if (this.key !== null && junctionKey !== this.key) this.since = s.tick;
    this.key = junctionKey;
    return this.ticks > 0 && s.tick - this.since < this.ticks;
  }
}

/**
 * Browser agent loop: one request in flight, the newest answer wins, and an
 * answer about a junction Pac-Man has already passed is dropped. The game
 * never waits for it.
 */
export class AgentLoop {
  private running = false;
  private controller: AbortController | null = null;
  private latency = new LatencyEstimate();
  /** Ask about the very next junction and let the game hold there until the answer arrives. */
  waitForModel = false;
  /** The junction an in-flight question is about. */
  pending: DecisionPoint | null = null;
  /** Off by default; see ReverseLock. */
  reverseLock = new ReverseLock(0);

  constructor(
    private hooks: AgentHooks,
    public policy: Policy,
    public encoder: Encoder,
  ) {}

  get isRunning(): boolean {
    return this.running;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    void this.run();
  }

  stop(): void {
    this.running = false;
    this.pending = null;
    this.controller?.abort();
  }

  private async run(): Promise<void> {
    let backoff = 500;
    while (this.running) {
      const game = this.hooks.getState();
      // Only ask while the game is moving: a frozen state repeats the same
      // request, which the server answers from cache and skews latency stats.
      if (game.phase !== "playing") {
        await sleep(50);
        continue;
      }
      const lookahead = this.waitForModel ? 0 : lookaheadTiles(game, this.latency.value);
      const dp = findDecisionPoint(game, lookahead);
      const encoded = this.encoder.encode(game, dp);
      const enc = this.reverseLock.blocked(game, findDecisionPoint(game).key) ? withoutBack(encoded) : encoded;
      if (enc.keys.length === 1) {
        const only = optionToInput(enc.keys[0], dp);
        if (only) this.hooks.apply(only);
        await sleep(33);
        continue;
      }
      const askedTick = game.tick;
      this.controller = new AbortController();
      // The player can be switched while a request is in flight; its answer then belongs to nobody.
      const policy = this.policy;
      this.pending = dp;
      try {
        const d = await policy.decide(enc, this.controller.signal, game);
        this.pending = null;
        if (policy !== this.policy) continue;
        backoff = 500;
        this.latency.update(d.latencyMs);
        const now = this.hooks.getState();
        const stale = now !== game || now.phase !== "playing" || !isStillAhead(now, dp, lookahead);
        const input = optionToInput(d.choice, dp);
        if (!stale && input) this.hooks.apply(input);
        this.hooks.onDecision({
          time: Date.now(),
          model: policy.name,
          encoder: this.encoder.name,
          junction: dp.key,
          keys: enc.keys,
          choice: d.choice,
          probabilities: d.probabilities,
          confidence: d.confidence,
          latencyMs: d.latencyMs,
          inputTokens: d.inputTokens,
          askedTick,
          answeredTick: now.tick,
          stale,
          state: enc.state,
        });
        // Let the queued input reach the game before planning the next
        // decision, so the lookahead sees the turn that was just planned.
        const applied = now.tick;
        for (let i = 0; i < 10 && this.hooks.getState().tick <= applied; i++) await sleep(5);
        if (d.latencyMs < 5) await sleep(33);
      } catch (err) {
        this.pending = null;
        if ((err as Error).name === "AbortError") return;
        if (policy !== this.policy) continue;
        this.hooks.onError(err);
        await sleep(backoff);
        backoff = Math.min(backoff * 2, 4000);
      }
    }
  }
}
