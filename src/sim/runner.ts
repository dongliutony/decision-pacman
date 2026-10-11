// Drives one seeded game with a policy, headless. Shared by the evaluation
// runner and the teacher data generator so both see the same game.
import { LatencyEstimate, lookaheadTiles, ReverseLock } from "../agent/loop.ts";
import type { Policy, PolicyDecision } from "../agent/policies.ts";
import { findDecisionPoint, isStillAhead, type DecisionPoint } from "../engine/decision.ts";
import { createGame, step, TPS } from "../engine/game.ts";
import type { GameState, StepInput } from "../engine/types.ts";
import { optionToInput, withoutBack, type EncodedDecision, type Encoder } from "../encoders/types.ts";

export type Clock = "realtime" | "lockstep";

export interface GameOptionsForRun {
  seed: number;
  policy: Policy;
  encoder: Encoder;
  clock: Clock;
  speed?: number;
  maxSeconds?: number;
  /** Optional second policy asked about every decision the player makes; its answer is not played. */
  labeler?: Policy;
  /** More encoders applied to the same state at every decision, so one run can train students that read different inputs. */
  alsoEncode?: Encoder[];
  onDecision?: (ev: DecisionEvent) => void;
  /** Game seconds after Pac-Man turns around or passes a junction during which "back" is not offered (see ReverseLock). 0 = off. */
  reverseLockSeconds?: number;
}

export interface DecisionEvent {
  seed: number;
  askedTick: number;
  answeredTick: number;
  dp: DecisionPoint;
  enc: EncodedDecision;
  /** The same state under each of `alsoEncode`, by encoder name. */
  alt?: Record<string, EncodedDecision>;
  decision: PolicyDecision | null;
  /** Labeler's answer; null if the labeler failed on this state. */
  label?: PolicyDecision | null;
  stale: boolean;
  error?: string;
}

export interface GameSummary {
  seed: number;
  score: number;
  pellets: number;
  ghostsEaten: number;
  deaths: number;
  level: number;
  secondsPlayed: number;
  decisions: number;
  failures: number;
  stale: number;
  latencyP50: number;
  latencyP90: number;
  endedBy: "gameover" | "time";
}

export function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

async function tryDecide(policy: Policy, enc: EncodedDecision, game: GameState): Promise<{ d: PolicyDecision | null; error?: string }> {
  try {
    return { d: await policy.decide(enc, undefined, game) };
  } catch (err) {
    return { d: null, error: String((err as Error).message ?? err) };
  }
}

export async function playGame(o: GameOptionsForRun): Promise<GameSummary> {
  const s: GameState = createGame(o.seed, { speed: o.speed ?? 1 });
  const maxTicks = (o.maxSeconds ?? 180) * TPS;
  const latencies: number[] = [];
  let decisions = 0;
  let failures = 0;
  let stale = 0;
  const estimate = new LatencyEstimate(o.clock === "realtime" ? 250 : 0);
  const lock = new ReverseLock(Math.round((o.reverseLockSeconds ?? 0) * TPS));

  const advance = (n: number, input?: StepInput | null) => {
    for (let i = 0; i < n && s.phase !== "gameover"; i++) step(s, i === 0 && input ? input : {});
  };

  while (s.phase !== "gameover" && s.stats.ticksPlaying < maxTicks) {
    if (s.phase !== "playing") {
      advance(1);
      continue;
    }
    const lookahead = o.clock === "realtime" ? lookaheadTiles(s, estimate.value) : 0;
    const dp = findDecisionPoint(s, lookahead);
    const encoded = o.encoder.encode(s, dp);
    const enc = lock.blocked(s, findDecisionPoint(s).key) ? withoutBack(encoded) : encoded;
    if (enc.keys.length === 1) {
      advance(1, optionToInput(enc.keys[0], dp));
      continue;
    }

    const askedTick = s.tick;
    const alt = o.alsoEncode?.length ? Object.fromEntries(o.alsoEncode.map((e) => [e.name, e.encode(s, dp)])) : undefined;
    const [played, labeled] = await Promise.all([
      tryDecide(o.policy, enc, s),
      o.labeler && o.labeler !== o.policy ? tryDecide(o.labeler, enc, s) : Promise.resolve(null),
    ]);
    const d = played.d;
    const label = o.labeler ? (o.labeler === o.policy ? d : (labeled?.d ?? null)) : undefined;

    let isStale = false;
    if (!d) {
      failures++;
      advance(1);
    } else {
      decisions++;
      latencies.push(d.latencyMs);
      estimate.update(d.latencyMs);
      const input = optionToInput(d.choice, dp);
      if (o.clock === "realtime") {
        advance(Math.max(0, Math.round((d.latencyMs / 1000) * TPS)));
        isStale = s.phase !== "playing" || !isStillAhead(s, dp, lookahead);
        if (isStale) stale++;
        advance(1, isStale ? null : input);
      } else {
        advance(1, input);
        // Ask again after passing the junction, or every 8 ticks to react to ghosts.
        for (let i = 0; i < 7 && s.phase === "playing" && findDecisionPoint(s).key === dp.key; i++) advance(1);
      }
    }
    o.onDecision?.({
      seed: o.seed,
      askedTick,
      answeredTick: s.tick,
      dp,
      enc,
      alt,
      decision: d,
      label,
      stale: isStale,
      error: played.error ?? labeled?.error,
    });
  }

  return {
    seed: o.seed,
    score: s.score,
    pellets: s.stats.pelletsEaten,
    ghostsEaten: s.stats.ghostsEaten,
    deaths: s.stats.deaths,
    level: s.level,
    secondsPlayed: Math.round(s.stats.ticksPlaying / TPS),
    decisions,
    failures,
    stale,
    latencyP50: Math.round(percentile(latencies, 0.5)),
    latencyP90: Math.round(percentile(latencies, 0.9)),
    endedBy: s.phase === "gameover" ? "gameover" : "time",
  };
}
