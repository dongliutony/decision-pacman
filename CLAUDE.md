# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The project rules, directory map, and environment live in `AGENTS.md`. Follow them; they are not repeated here.

@AGENTS.md

## Language and repo setup

- Chat with the user in Chinese (中文).
- Write everything that goes into a file in English: code, comments, docs, commit messages, PR text, and memory files.
- The user works in their own fork for learning, research, and improvements. `origin` is the fork (`dongliutony/decision-pacman`); `upstream` is the original repo (`grapeot/decision-pacman`), used only to fetch updates. Never push to or open PRs against `upstream`.
- The branch `local/learning-poc` holds this work.
- `docs/local-usage.md` is the how-to for this branch (hosted Jev, the local model, random players). Keep it current when those commands change.

## Commands

```bash
npm run dev                      # game at http://localhost:5173 (needs Ollama, or ?model=... for another player)
npm run dev:site                 # the GitHub Pages trailer site (site/), built by build:site into dist-site/
npm run typecheck                # tsc --noEmit; CI runs this, then npm test
npm test                         # Vitest, offline, tests/**/*.test.ts
npx vitest run tests/engine.test.ts          # one file
npx vitest run tests/engine.test.ts -t "tunnel"   # one test by name
npm run build                    # typecheck + vite build of the game

# Headless seeded games (writes JSONL under runs/, gitignored). Evaluation protocol: docs/guides/evaluate.md
npm run headless -- --policy greedy --games 10 --seed 100 --max-seconds 300
npm run headless -- --policy tev1:4b --encoder features --clock realtime --games 10 --seed 100 --max-seconds 300

npm run gen-data                 # teacher-labeled training states (refuses evaluation seeds)
npm run record -- --model tev1:4b --seconds 30 --seed 100 --out docs/media/demo.mp4

# Python tests (stdlib; MLX test skips off Apple silicon)
cd training && python3 -m unittest test_build_sft
cd training && python -m unittest test_train_mlx
```

Hosted players need keys in the environment: `jev` needs `TYPESAFE_API_KEY`, `openai:<model>` needs `OPENAI_API_KEY`, and `teacher*` policies need `TEACHER_BASE_URL` and `TEACHER_MODEL` (see `.env.example`).

## Architecture

- **Two loops, never coupled.** `src/engine/` is a pure `createGame(seed, options)` / `step(state, intent)` over plain data at a fixed 30 Hz. The agent loop (`src/agent/loop.ts`) runs asynchronously: it picks the next junction ahead (pushed further out by a moving average of latency), encodes the state, queries a player, and turns the answer into `StepInput.turn` (planned turn at that junction) or `StepInput.reverse` ("back"). An answer whose junction is no longer ahead is dropped. Keyboard play uses `StepInput.intent`.
- **One engine, three hosts.** The browser (`src/main.ts` + `src/render/canvas.ts`), the headless runner (`scripts/run_headless.ts` via `src/sim/runner.ts`), and data generation all import the same engine and encoders. The headless `realtime` clock converts measured latency into elapsed ticks (the evaluation clock); `lockstep` pauses the game per answer. The browser's `?wait=1` is the lockstep equivalent.
- **Encoders** (`src/encoders/`) return `{ state, instructions, options }`. `features` is the default (~200 tokens of compact JSON; tokens are latency). `features-peek5s` (`peek.ts`) includes lookahead and is only for labeling and recording.
- **Policy names select the player** (`src/agent/factory.ts`): Ollama decision models over `/v1/systemone` (`client.ts`), `llm:<model>` chat models over `/api/chat` with a JSON schema (`llm.ts`), `jev` hosted, `openai:<model>` Decisions API (`decisions.ts`), `teacher*` chat teacher (`teacher.ts`), `oracle-*` rollout search (`oracle.ts`), plus `random` / `greedy`. In dev, Vite proxies `/decide` and `/openai` and injects the API keys server-side (`vite.config.ts`), so keys never reach the page.
- **Prompt parity is load-bearing.** The decision prompt Ollama builds is reproduced byte for byte in `training/prompt.py` (training) and `src/agent/prompt.ts` (iOS, via `src/agent/native.ts`). Changing the encoder output or prompt shape changes what the trained model sees; keep all three in sync and keep the fixture tests passing.
- **Distillation pipeline:** headless engine states → `teacher-peek5s` labels (teacher sees `oracle.ts` rollouts) → `training/build_sft.py` → LoRA on `Qwen/Qwen3.5-0.8B` (`train.py` on CUDA, `train_mlx.py` on Mac) → GGUF (`export_gguf*.sh`) → Ollama → scored with `training/score_ollama.py` and headless games.
- **Seed ranges:** evaluation 100-199, validation 900-999, training 1000+.
