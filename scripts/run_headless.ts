// Plays seeded games without rendering and writes one JSONL record per decision.
//
//   npm run headless -- --policy tev1:4b --games 5 --seed 100 --clock realtime
//   npm run headless -- --policy greedy --games 20
//
// realtime: the model's measured latency turns into game ticks that pass before
//           the answer applies, as in the browser.
// lockstep: the game waits for every answer (an upper bound on decision quality).
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadDotEnv, makePolicy, policyEnvFromProcess } from "../src/agent/factory.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import { playGame, type Clock, type GameSummary } from "../src/sim/runner.ts";

loadDotEnv();

const { values: args } = parseArgs({
  options: {
    policy: { type: "string", default: "tev1:4b" },
    "base-url": { type: "string" },
    encoder: { type: "string", default: "features" },
    clock: { type: "string", default: "realtime" },
    games: { type: "string", default: "3" },
    seed: { type: "string", default: "100" },
    speed: { type: "string", default: "1" },
    "max-seconds": { type: "string", default: "180" },
    // Game seconds after Pac-Man turns around or passes a junction during which "back" is not offered. 0 = off.
    "reverse-lock": { type: "string", default: "0" },
    out: { type: "string", default: "runs" },
  },
});

const clock = args.clock as Clock;
if (clock !== "realtime" && clock !== "lockstep") throw new Error("--clock must be realtime or lockstep");

async function main() {
  const env = policyEnvFromProcess();
  if (args["base-url"]) env.decisionBaseUrl = args["base-url"];
  const games = Number(args.games);
  const seed0 = Number(args.seed);
  const reverseLockSeconds = Number(args["reverse-lock"]);
  const lockTag = reverseLockSeconds > 0 ? `_lock${reverseLockSeconds}s` : "";
  const tag = `${new Date().toISOString().replace(/[:.]/g, "-")}_${args.policy!.replace(/[^\w.-]/g, "_")}_${args.encoder}_${clock}${lockTag}_p${process.pid}`;
  const dir = join(args.out!, tag);
  mkdirSync(dir, { recursive: true });
  const logPath = join(dir, "decisions.jsonl");
  writeFileSync(logPath, "");

  const summaries: GameSummary[] = [];
  for (let g = 0; g < games; g++) {
    const seed = seed0 + g;
    const policy = makePolicy(args.policy!, seed, env);
    const encoder = ENCODERS[policy.requiresEncoder ?? args.encoder!];
    if (!encoder) throw new Error(`unknown encoder ${args.encoder}`);
    const sum = await playGame({
      seed,
      policy,
      encoder,
      clock,
      speed: Number(args.speed),
      maxSeconds: Number(args["max-seconds"]),
      reverseLockSeconds,
      onDecision: (ev) =>
        appendFileSync(
          logPath,
          JSON.stringify({
            seed,
            askedTick: ev.askedTick,
            answeredTick: ev.answeredTick,
            junction: ev.dp.key,
            keys: ev.enc.keys,
            choice: ev.decision?.choice ?? null,
            probabilities: ev.decision?.probabilities,
            confidence: ev.decision?.confidence,
            latencyMs: ev.decision ? Math.round(ev.decision.latencyMs) : null,
            inputTokens: ev.decision?.inputTokens,
            stale: ev.stale,
            error: ev.error,
            state: ev.enc.state,
          }) + "\n",
        ),
    });
    summaries.push(sum);
    console.log(
      `seed ${seed}: score ${sum.score}, pellets ${sum.pellets}/244, ghosts ${sum.ghostsEaten}, deaths ${sum.deaths}, ` +
        `level ${sum.level}, ${sum.secondsPlayed}s, ${sum.decisions} decisions (${sum.stale} stale, ${sum.failures} failed), p50 ${sum.latencyP50} ms`,
    );
  }
  const mean = (k: keyof GameSummary) => summaries.reduce((a, s) => a + (s[k] as number), 0) / summaries.length;
  const totalDecisions = summaries.reduce((a, s) => a + s.decisions, 0);
  const overall = {
    policy: args.policy,
    encoder: args.encoder,
    clock,
    speed: Number(args.speed),
    games,
    seed0,
    reverseLockSeconds,
    meanScore: Math.round(mean("score")),
    meanPellets: Math.round(mean("pellets") * 10) / 10,
    meanSeconds: Math.round(mean("secondsPlayed")),
    meanLatencyP50: Math.round(mean("latencyP50")),
    staleRate: Math.round((summaries.reduce((a, s) => a + s.stale, 0) / Math.max(1, totalDecisions)) * 1000) / 1000,
  };
  writeFileSync(join(dir, "summary.json"), JSON.stringify({ overall, games: summaries }, null, 2));
  console.log(JSON.stringify(overall));
  console.log(`wrote ${dir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
