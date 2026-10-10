import type { EncodedDecision } from "../encoders/types.ts";
import { TPS } from "../engine/game.ts";
import type { Dir, GameState } from "../engine/types.ts";
import { askSystemOne, type SystemOneConfig } from "./client.ts";

export interface PolicyDecision {
  choice: string;
  probabilities?: Record<string, number>;
  /** Per-option values from search policies (higher is better; not probabilities). */
  values?: Record<string, number>;
  confidence?: number;
  inputTokens?: number;
  latencyMs: number;
}

export interface Policy {
  name: string;
  /** Encoder this policy expects, when it reads the encoded state itself. */
  requiresEncoder?: string;
  /** `game` is the live state, for policies that search the engine instead of reading the encoding. */
  decide(enc: EncodedDecision, signal?: AbortSignal, game?: GameState): Promise<PolicyDecision>;
}

export function systemOnePolicy(cfg: SystemOneConfig): Policy {
  return {
    name: cfg.model,
    async decide(enc, signal) {
      const a = await askSystemOne(cfg, enc, signal);
      return a;
    },
  };
}

function seededRandom(seed: number): () => number {
  let t = seed | 0;
  return () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform over the legal options, with its own seeded generator. */
export function randomPolicy(seed = 1): Policy {
  const rand = seededRandom(seed);
  return {
    name: "random",
    async decide(enc) {
      return { choice: enc.keys[Math.floor(rand() * enc.keys.length)], latencyMs: 0 };
    },
  };
}

/**
 * Random walk without dithering: one uniform exit per junction, kept while Pac-Man heads there,
 * and once every `reverseEveryTicks` a draw that turns back with probability `backChance`
 * (default 1/4, so going on and turning back are 3:1).
 * `random` re-rolls "back" on every question, so it turns around many times a second.
 */
export function randomWalkPolicy(seed = 1, reverseEveryTicks = 30, name = "random-1s", backChance = 0.25): Policy {
  const rand = seededRandom(seed);
  let lastFlip = -Infinity;
  let planned: { junction: string; choice: Dir } | null = null;
  return {
    name,
    async decide(enc, _signal, game) {
      const tick = game?.tick ?? 0;
      if (tick < lastFlip) lastFlip = -Infinity; // a new game started
      const exits = enc.keys.filter((k) => k !== "back");
      if (enc.keys.includes("back") && (exits.length === 0 || tick - lastFlip >= reverseEveryTicks)) {
        lastFlip = tick;
        if (exits.length === 0 || rand() < backChance) {
          planned = null;
          return { choice: "back", latencyMs: 0 };
        }
      }
      const junction = enc.decision.key;
      if (planned?.junction !== junction || !exits.includes(planned.choice)) {
        planned = { junction, choice: exits[Math.floor(rand() * exits.length)] };
      }
      return { choice: planned.choice, latencyMs: 0 };
    },
  };
}

/** "random", or "random-<seconds>s" for the random walk; null for any other name. */
export function randomByName(name: string, seed: number): Policy | null {
  if (name === "random") return randomPolicy(seed);
  const walk = /^random-(\d+(?:\.\d+)?)s$/.exec(name);
  return walk ? randomWalkPolicy(seed, Math.round(Number(walk[1]) * TPS), name) : null;
}

interface FactsLike {
  ghost: number | null;
  edible?: number | null;
  pellets: number;
  food: number | null;
}

/**
 * Scripted baseline over the features encoder: chase edible ghosts nearby,
 * flee when a normal ghost is close, otherwise head for the nearest pellet.
 */
export function greedyChoice(enc: EncodedDecision, dangerSteps = 4): string {
  const options = (enc.state as { options: Record<string, FactsLike> }).options;
  const keys = enc.keys.filter((k) => options[k]);
  const far = (v: number | null | undefined) => (v === null || v === undefined ? 99 : v);
  const edible = keys.filter((k) => far(options[k].edible) <= 8 && far(options[k].ghost) > dangerSteps);
  if (edible.length > 0) {
    edible.sort((a, b) => far(options[a].edible) - far(options[b].edible));
    return edible[0];
  }
  const safe = keys.filter((k) => far(options[k].ghost) > dangerSteps);
  if (safe.length === 0) {
    const byGhost = [...keys].sort((a, b) => far(options[b].ghost) - far(options[a].ghost));
    return byGhost[0];
  }
  safe.sort((a, b) => far(options[a].food) - far(options[b].food) || options[b].pellets - options[a].pellets);
  return safe[0];
}

export function greedyPolicy(dangerSteps = 4): Policy {
  return {
    name: "greedy",
    requiresEncoder: "features",
    async decide(enc) {
      return { choice: greedyChoice(enc, dangerSteps), latencyMs: 0 };
    },
  };
}
