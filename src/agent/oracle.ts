// Search baseline that needs no model: for each option, copy the game, play
// the option, and roll the engine forward with greedy play at later
// junctions. Ghosts follow the real rules, so the lookahead is exact except
// for frightened ghosts' random turns.
import { distancesFrom, findDecisionPoint } from "../engine/decision.ts";
import { step, TPS } from "../engine/game.ts";
import { COLS, Food, ROWS, index, wrapX } from "../engine/maze.ts";
import type { GameState } from "../engine/types.ts";
import { featuresEncoder } from "../encoders/features.ts";
import { optionToInput, type EncodedDecision } from "../encoders/types.ts";
import { greedyChoice, type Policy, type PolicyDecision } from "./policies.ts";

export interface OracleOptions {
  /** Rollout length in game seconds. */
  horizonSeconds?: number;
  /** Rollouts per option with different random streams for frightened ghosts. */
  samples?: number;
  /** What a rollout earns: game points ("score", the default), or 10 per pellet eaten ("pellets"). */
  objective?: Objective;
}

/**
 * "score" counts every game point, so eating frightened ghosts (200-1600) and power pellets (50) pulls the search.
 * "pellets" counts 10 per pellet of either kind, the unit the ladder measures; the death penalty stays, because
 * a dead Pac-Man eats nothing and the third death ends the game.
 */
export type Objective = "score" | "pellets";

const DEATH_PENALTY = 100_000;

function clone(s: GameState): GameState {
  return structuredClone(s);
}

export function nearestFood(s: GameState): number {
  const dist = distancesFrom(wrapX(Math.round(s.pac.x)), Math.round(s.pac.y));
  let best = 99;
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const i = index(x, y);
      if (s.food[i] !== Food.None && dist[i] >= 0 && dist[i] < best) best = dist[i];
    }
  }
  return best;
}

/** Plays `key` now, then greedy play at later junctions, for up to `horizonTicks`. Returns the end state. */
export function rollout(s0: GameState, enc: EncodedDecision, key: string, horizonTicks: number, rngSeed: number): GameState {
  const s = clone(s0);
  s.rng = (s.rng ^ Math.imul(rngSeed + 1, 0x9e3779b1)) | 0;
  const first = optionToInput(key, enc.decision);
  step(s, first ?? {});
  let decided = enc.decision.key;
  for (let t = 1; t < horizonTicks; t++) {
    if (s.phase !== "playing") break;
    let input = {};
    const dp = findDecisionPoint(s);
    if (dp.key !== decided && dp.distance < 1.5) {
      const e = featuresEncoder.encode(s, dp);
      const choice = e.keys.length === 1 ? e.keys[0] : greedyChoice(e);
      input = optionToInput(choice, dp) ?? {};
      decided = dp.key;
    }
    step(s, input);
  }
  return s;
}

/** What happened in a rollout, as facts a model can read. */
export interface PeekOutcome {
  /** Seconds until Pac-Man dies, or null if he survives the whole horizon. */
  dies_after_s: number | null;
  pellets: number;
  power_pellets: number;
  ghosts_eaten: number;
  points: number;
}

export function peek(s0: GameState, enc: EncodedDecision, key: string, horizonTicks: number): PeekOutcome {
  const s = rollout(s0, enc, key, horizonTicks, 0);
  const died = s.stats.deaths > s0.stats.deaths;
  const eaten = s.stats.pelletsEaten - s0.stats.pelletsEaten;
  const power = Math.max(0, countPower(s0) - countPower(s));
  return {
    dies_after_s: died ? Math.round(((s.stats.ticksPlaying - s0.stats.ticksPlaying) / TPS) * 10) / 10 : null,
    pellets: eaten - power,
    power_pellets: power,
    ghosts_eaten: s.stats.ghostsEaten - s0.stats.ghostsEaten,
    points: s.score - s0.score,
  };
}

function countPower(s: GameState): number {
  let n = 0;
  for (const f of s.food) if (f === Food.Power) n++;
  return n;
}

/** Value of playing `key` now: score gained over the horizon, minus a large penalty for dying (less if later). */
export function rolloutValue(
  s0: GameState,
  enc: EncodedDecision,
  key: string,
  horizonTicks: number,
  rngSeed: number,
  objective: Objective = "score",
): number {
  const s = rollout(s0, enc, key, horizonTicks, rngSeed);
  const gained = objective === "pellets" ? 10 * (s.stats.pelletsEaten - s0.stats.pelletsEaten) : s.score - s0.score;
  if (s.stats.deaths > s0.stats.deaths) {
    // Dying later is better than dying sooner.
    return -DEATH_PENALTY + (s.stats.ticksPlaying - s0.stats.ticksPlaying) * 10 + gained;
  }
  return gained - nearestFood(s);
}

export function oraclePolicy(opts: OracleOptions = {}): Policy {
  const horizonTicks = Math.round((opts.horizonSeconds ?? 5) * TPS);
  const samples = opts.samples ?? 1;
  const objective = opts.objective ?? "score";
  return {
    name: `oracle-${objective === "pellets" ? "pellets-" : ""}${opts.horizonSeconds ?? 5}s`,
    async decide(enc, _signal, game): Promise<PolicyDecision> {
      if (!game) throw new Error("the oracle needs the live game state");
      const started = performance.now();
      const values: Record<string, number> = {};
      let best = enc.keys[0];
      let bestValue = -Infinity;
      for (const key of enc.keys) {
        let total = 0;
        for (let i = 0; i < samples; i++) total += rolloutValue(game, enc, key, horizonTicks, i, objective);
        const value = total / samples;
        values[key] = Math.round(value * 10) / 10;
        if (value > bestValue) {
          bestValue = value;
          best = key;
        }
      }
      return { choice: best, latencyMs: performance.now() - started, values };
    },
  };
}
