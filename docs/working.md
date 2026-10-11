# Working notes

## Changelog

### 2026-09-30

- Scaffolded the project: docs, AGENTS.md, .gitignore, .env.example, and empty src/scripts/tests.
- Renamed the project from nimble_pacman to decision_pacman after the default model moved to tev1:4b.
- Rewrote the PRD and RFC around four stages (real-time game, teacher data, fine-tuning a 0.8B model, evaluation), and moved measurements into `docs/model_evaluation.md`.
- Verified the local Ollama 0.35.0 `/v1/systemone` endpoint with `nimble`, `tev1:4b`, and `tev1:0.8b` on an M3 Ultra.
- Measured latency for the full ASCII board (~500 tokens: nimble ~430 ms, tev1:4b ~240 ms, tev1:0.8b ~80 ms) and for per-direction feature JSON (~200 tokens: nimble ~220–260 ms, tev1:4b ~150–170 ms, tev1:0.8b ~55–75 ms).
- Confirmed that Ollama returns CORS headers for `http://localhost:*` origins, so the browser can call it directly.
- Wrote the PRD, RFC, and test strategy.
- Added `scripts/probe_decision_models.py`: 40 fresh synthetic scenarios with a known correct move per model. Results on json state: nimble q8 270 ms / 0.68, tev1:4b 156 ms / 0.88, tev1:0.8b 67 ms / 0.42, laya:en (ollaya) 20 ms / 0.40.
- Quantization variants on the same probe: nimble:9b-q4_K_M 286 ms / 0.60, tev1:4b-q4_K_M 161 ms / 0.88, tev1:0.8b-bf16 65 ms / 0.40. q4 buys no speed on M3 Ultra, because short-output decisions are prefill-bound (compute), not bandwidth-bound.
- Tested a plain-English state against compact JSON: no gain (laya 0.40 to 0.47, tev1:4b 0.88 to 0.78). Kept JSON.
- Installed ollaya 0.7.5 (port 11435) to benchmark Kev, decider, laya, and winnow. kev:0.8b ran at 729 ms / 0.40 on the Mac (CPU path).
- Confirmed Tev1 is a plain LoRA on Qwen3.5 with an MIT-licensed recipe. Its 4B weights are on Hugging Face. The 0.8B weights have been seen only as Ollama GGUF, so fine-tuning starts from the Qwen3.5-0.8B base.
- Added `scripts/probe_teacher.py` and measured Qwen3.8-27B (NVFP4, OpenAI-compatible server) as a possible distillation teacher on the same scenarios. Thinking off: p50 270 ms, 40/40 parseable, 40/40 legal, accuracy 0.97. Thinking on: p50 1.1 s, accuracy 1.00, ~190 output tokens. Thinking off at concurrency 8: 16 labels/s.

### 2026-09-30 (Stage 1)

- Built the engine (`src/engine/`): classic maze, fixed 30 Hz step, four ghost targeting rules, scatter/chase schedule, frightened mode, ghost house, tunnel, lives, and level clear. Deterministic from a seed.
- Added decision points (`findDecisionPoint`): the next junction on the current heading, following single-exit corners automatically.
- Added the `features` and `ascii-window` encoders, the `/v1/systemone` client, random/greedy/model policies, and the browser agent loop.
- Added the canvas renderer and HUD (probability bars, latency, stale rate, ticks per second, the encoded state the model sees) and keyboard play.
- Added `scripts/run_headless.ts` (realtime and lockstep clocks, JSONL logs under `runs/`) and `scripts/record_demo.ts` (Playwright recording converted to MP4).
- Headless, 1x speed, realtime clock, seeds 100+: random 30.3 pellets (10 games), greedy 200.7 (10 games), tev1:4b 168.7 (3 games, p50 207 ms, 10% stale).
- Recorded `docs/media/demo_tev1_4b.mp4` (30 s, seed 100): 1250 points and 101 pellets, one life lost near the end.
- Tests: 28 passing (engine, planned turns, decision points, encoders).

### 2026-09-30 (Stage 2 smoke test)

- Moved the game-driving loop into `src/sim/runner.ts`, shared by the headless runner and data generation. Greedy results were identical before and after (10 games, 200.7 pellets).
- Added the teacher policy (`src/agent/teacher.ts`): an OpenAI-compatible chat model sees the student's encoded state and answers `{"move", "reason"}`. An unparseable or illegal answer drops the state.
- Added `scripts/gen_teacher_data.ts`: a player drives the game in lockstep and a labeler labels every decision state. Records use schema `decision-pacman/teacher-v1` under `data/` (gitignored). Seeds 100-199 are refused.
- Smoke test (player and labeler both Qwen3.8-27B, thinking off, seed 1000, 30 s of game time): 144 labeled states in 46 s wall time, 0 failures, label latency p50 301 ms and p90 368 ms, 3.1 labels per second sequential. The teacher scored 2470 with 155 pellets and 1 death in those 30 s. Labels: left 53, up 35, down 25, right 22, back 9. It agreed with the greedy rules on 70% of states. The disagreements sampled were ties, which the teacher broke toward a power pellet.
- Tests: 33 passing (adds teacher reply parsing and prompt tests).

### 2026-09-30 (teacher comparison and search baseline)

- Added the `ascii-full` encoder: the whole board with row and column numbers, plus positions in words (about 1,350 characters).
- Qwen3.8-27B as a player, lockstep, 30 s, seeds 1001-1003: features, thinking off: 169 pellets and 2 deaths in total. Features, thinking on: 176 pellets, 0 deaths, 13 unparseable answers out of about 500. ASCII full map, thinking off: 99 pellets, 6 deaths. ASCII full map with thinking was stopped after 48 minutes, too slow to matter.
- Qwen misreads the ASCII map. Sampled reasons mention dead ends (the maze has none), a tunnel on row 29 (it is on row 14), and a far power pellet when one was on the same row.
- Added `src/agent/oracle.ts`: for each option, copy the game, play the option, and roll the real engine forward with greedy play at later junctions. The score gained is penalized heavily for a death, less for a later one.
- Oracle results on evaluation seeds 100-109, realtime, 5-minute cap: `oracle-3s` averaged 1,009 pellets (about 4 levels) and 16,546 points, 3 ms per decision. `oracle-5s` averaged 1,109 pellets and 19,789 points, 5 ms. `oracle-8s` averaged 1,011 pellets and 15,858 points, 7 ms. All 30 games reached the time cap. A longer horizon is not better past 5 s, probably because the greedy rollout policy gets less reliable further out.
- 5-minute lockstep head-to-head, seeds 100-102: `oracle-3s` ate 971, 1,178, and 1,211 pellets with 1 death per game. Qwen (features, thinking off) ate 361, 361, and 180 pellets and lost all three lives within 74, 93, and 39 seconds.
- A literature survey agreed: the strongest simple agents rely on an exact ghost model and a safe-path test, which this engine provides for free.
- Fixed output directories colliding when two runs start in the same millisecond. Names now include the encoder and the process id.

### 2026-09-30 (Stage 3: fine-tuning)

- Generated v1 data with `oracle-5s` labels: 57,249 training and 8,050 validation states. After removing duplicates: 31,585 and 4,975. Oracle, greedy, and random players, train seeds 1000-3029, validation seeds 900-944.
- Found how Ollama builds System One prompts (`decision/systemone.go`) and reproduced it in `training/prompt.py`.
- Verified early that a raw GGUF imported through `/api/create` with Tev1's system prompt serves `/v1/systemone` with probabilities identical to `tev1:0.8b`.
- Trained a LoRA on `Qwen/Qwen3.5-0.8B` on one borrowed RTX 5090 (2 epochs, 1,975 steps, about 40 minutes). Validation agreement with the oracle went from 28.8% to 52.7%.
- Exported Q8_0 GGUF, imported it into Ollama as `pacman-0.8b`, and returned the GPU to the cluster.
- Parity: HF bf16 against Ollama Q8_0 on 50 states, median probability difference 0.010, max 0.027.
- Offline agreement (1,000 validation states): `tev1:0.8b` 31.8%, `tev1:4b` 38.4%, `pacman-0.8b` 53.9%.
- In-game (seeds 100-109, realtime): `tev1:0.8b` 69 pellets, `tev1:4b` 148, greedy 201, `pacman-0.8b` 456 (8 of 10 games cleared level 1), `oracle-5s` 1,109.
- Recorded `docs/media/demo_pacman_08b.mp4` (30 s, seed 100): 2,660 points, 170 pellets, no lives lost.

### 2026-09-30 (wrap-up)

- Recorded `docs/media/demo_tev1_08b.mp4` (30 s, seed 100): 710 points, 71 pellets, two lives lost. The three clips on the same seed now cover the teaching sequence, and the README compares them.
- Stopped here by decision. The follow-ups (richer encoder, DAgger, full-map student) are listed in `docs/model_evaluation.md` but not planned.
- The fine-tuned GGUF (`pacman-0.8b`, Q8_0, about 800 MB) and the training data stay outside the repository.

### 2026-09-30 (iOS app)

- Checked feasibility on an iPhone 16 Pro Max (A18 Pro). llama.cpp's Metal backend runs every Qwen3.5 layer, including Gated DeltaNet, on the GPU: 26/26 layers offloaded, 2 graph splits. Decision latency p50 380 ms, p90 431 ms. The results matched Ollama on 60 of 60 decisions, with a maximum probability difference of 0.0003. First model load took 17 s (shader compilation); later loads took 0.37 s.
- Added `ios/`: a SwiftUI app that bundles the web build in a WKWebView served from `app://local/` and answers the page's decisions with llama.cpp on device through a `WKScriptMessageHandlerWithReply` bridge. A Latency tab keeps the benchmark, `decisionpacman://bench?run_id=` runs it, and the page writes a status heartbeat to `Documents/game_status.json`.
- Added `src/agent/prompt.ts` (the Ollama System One prompt in TypeScript, byte-identical to the HF chat template on 20 fixtures) and `src/agent/native.ts` (the bridge policy). Added swipe controls, a canvas that fits narrow screens, and relative asset paths.
- On device the game ran at 30 ticks per second with decisions at p50 420-440 ms through the bridge, and ate 227 pellets in the first 75 s. 27% of answers arrived after Pac-Man had passed their junction (13% on the Mac).
- Reported issue: during on-device inference the whole game appeared to stall, ghosts included. Investigated in the next entry; not reproduced.

### 2026-09-30 (iOS stall investigation)

- Instrumented the status heartbeat (every 2 s): `seq`, `mode`, `phase`, `paused`, decision count, stale count, latency and input-token medians over the last 50 decisions, the device thermal state, and a frame window with animation frames, simulation ticks, the largest gap between frames, and simulated time dropped by the catch-up cap. Added `decisionpacman://control?mode=ai|human&paused=0|1&speed=<x>&restart=1`, which calls the page's new `window.__pacman.setMode/setPaused/setSpeed/restart`, so a run can be driven and compared from a Mac with `devicectl`.
- Measured on the iPhone 16 Pro Max, pacman-0.8b Q8_0 on Metal, features encoder, 1x speed. AI mode, 99 two-second windows in the playing phase over four games: frames median 121 (min 119), ticks median 60 (min 60), largest frame gap median 19 ms (max 47 ms), dropped simulated time 0 ms. All 225 AI windows, including dying, ready, and game over: ticks 59-61, largest gap at most 55 ms, 0 ms dropped. Human mode (no inference), 16 playing windows: frames median 121 (min 120), ticks 60, largest gap median 17.5 ms (max 20 ms), dropped 0 ms. The GPU work of back-to-back decisions does not slow the page's frame loop or the simulation.
- The stall was therefore not a world-wide freeze. What looks like one is either the designed freezes (1.5 s dying plus 2 s ready after each death, when ghosts stop too) or Pac-Man waiting at a junction whose straight-ahead exit is a wall because the answer was late, while the ghosts keep moving. The user confirmed the second is the intended behavior. No game or inference setting changed.
- Decision latency rises within a game: p50 (last 50 decisions) about 360-375 ms at the start, 650-850 ms after 40-60 s of continuous inference, back to about 430 ms after a 40 s rest. Input tokens stay flat (p50 344-382 in the game where they were logged), so the prompt is not growing. The device reported thermal state `serious` throughout the two games where it was logged, so throttling under sustained GPU load is the likely cause. As latency climbs, decisions fall from about 4.5 to 2.5 per 2 s, and the stale rate per game was 23%, 34%, 45%, and 35%, so Pac-Man misses more turns late in a game.

### 2026-09-30 (music and sound)

- Added original chiptune music and sound effects, synthesized with Web Audio (`src/audio/`). Music follows the game phase; effects follow game events. M or the Sound button mutes.
- Offline render of a 40 s tour (`npm run render-audio`): -21 LUFS integrated, peak -3.7 dBFS.
- The iOS app uses an ambient audio session, so Silent Mode mutes the game. A playback session that ignores Silent Mode was tried and dropped: a game should stay quiet when the phone is silenced.

### 2026-09-30 (Jev)

- Connected TypeSafe's hosted Jev (`--policy jev`, `TYPESAFE_API_KEY`). Same `/v1/systemone` format as Ollama; the API rejects `keep_alive`, so hosted requests leave it out.
- 10 evaluation games, realtime, 1x: mean 178 pellets, 2,482 points, 52 s, 0 levels cleared, p50 115 ms, 3% stale. Agreement with the oracle on 1,000 validation states: 39.9% (decisive 40.9%), cross-entropy 1.41.
- Recorded `docs/media/demo_jev.mp4` (30 s, seed 100): 900 points, 86 pellets, one life lost. Browser p50 was about 210 ms through the dev proxy while other runs shared the API.
- Jev adds about 300 input tokens of its own per request (325 for a trivial state).

### 2026-09-30 (Qwen with thinking and lookahead)

- Split the oracle's rollout into `rollout()` and added `peek()`, which reports a rollout as facts (seconds until death, pellets, power pellets, ghosts eaten, points).
- Added `teacher-peek<N>s` (and `teacher-think-peek<N>s`): the Qwen teacher gets one N-second simulated future per option in its prompt.
- Lockstep, 5-minute cap: Qwen with 5 s lookahead (thinking off) averaged 1,396 pellets and 22,054 points over seeds 100-109, all games reaching level 6, p50 309 ms. Qwen with thinking and no lookahead averaged 531 pellets over seeds 100-102 and lost all lives within 97-131 s, p50 2.4 s.

### 2026-09-30 (plain small LLMs)

- Added `llm:<model>` (`src/agent/llm.ts`): a stock Ollama chat model gets the teacher prompt over the features state, without the reason field, through `/api/chat` with a JSON schema whose `move` is an enum of the legal options. Thinking off, temperature 0, `keep_alive: -1`. Probabilities come from the top-10 logprobs at the move token, renormalized over the options. `teacherPrompt` now takes the answer line as a parameter.
- Added `--chat` to `training/score_ollama.py`, which asks the same way. Its prompt matched `teacherPrompt` byte for byte on 600 validation states.
- Pulled `qwen3.5:4b`, `gemma4:e4b`, and `phi4-mini` (all Q4_K_M). Latency probe on 40 fresh states, M3 Ultra: p50 263 ms, 243 ms, and 170 ms, about 300 input tokens and 7 output tokens per answer.
- 10 evaluation games each, realtime, 1x, 5-minute cap: `qwen3.5:4b` 197 pellets, 2,553 points, 47 s, p50 267 ms, 9% stale; `gemma4:e4b` 230 pellets, 3,276 points, 55 s, p50 253 ms, 4% stale, 1 level cleared; `phi4-mini` 232 pellets, 3,142 points, 55 s, p50 187 ms, 4% stale, 2 levels cleared. No failed answers in 5,950 decisions. Jev was 178 pellets.
- Oracle agreement on 1,000 validation states: `qwen3.5:4b` 45.4% (cross-entropy 1.67), `gemma4:e4b` 43.6%, `phi4-mini` 44.0% (1.72). Jev was 39.9%, `tev1:4b` 38.4%.

### 2026-09-30 (cost of specialization)

- Added `scripts/eval_general_decisions.py`: scores models through `/v1/systemone` on JevBench's public items (MIT, pinned commit, hash-checked, downloaded to `data/jevbench/`). Items over 4,000 characters are skipped because the loaded models have a 2,050-token context: 194 of 231 items remain (48 easy, 72 standard, 74 hard).
- Ollama 0.35.0, M3 Ultra, one pass per model. Accuracy (probability on the correct option), p50 latency: `tev1:4b` 0.81 (0.79), 196 ms; `tev1:0.8b` 0.66 (0.63), 68 ms; `pacman-0.8b` 0.49 (0.36), 69 ms. Chance is 0.32. By tier, `pacman-0.8b` against `tev1:0.8b`: easy 0.79 vs 1.00, standard 0.39 vs 0.69, hard 0.39 vs 0.41.
- `pacman-0.8b` favors the first two options (166 of 194 picks, against 120 correct) and says yes on 85% of yes/no items (labels: 47%).
- Same-day rerun of the Pac-Man probe, accuracy (probability on the correct move): `tev1:4b` 0.88 (0.64), `tev1:0.8b` 0.42 (0.40), `pacman-0.8b` 0.93 (0.52). The probe repeats its seed-42 scenarios, so its latency is not reported here; earlier runs of the same states could be cached.
- JevBench's published per-item results for Jev 1.13.0 give 0.89 on the same 194 items.

### 2026-09-30 (lookahead for a closed model)

- Added the `features-peek5s` encoder: the features encoding plus each option's `peek()` outcome under `lookahead_5s`, and one paragraph of instructions explaining it. Any `/v1/systemone` model can now receive the lookahead Qwen saw in its prompt.
- Jev (TypeSafe's hosted API), seeds 100-109, realtime, 1x, 5-minute cap: 567 pellets, 9,150 points, 175 s, mean level 2.5, p50 111 ms, 2.2% stale, 865 input tokens per decision. Features only: 178 pellets, 52 s, 609 tokens. Every game still lost all three lives. 9.72 million input tokens, about $0.41.
- `tev1:4b` (local Ollama, M3 Ultra), same setup: 212 pellets, 2,907 points, 68 s, p50 260 ms, 8.6% stale, 578 tokens. Features only: 148 pellets, 202 ms, 368 tokens.
- The input knob works for a closed model, but its value depends on the model: +43% for `tev1:4b`, about +220% for Jev, about +360% for Qwen (lockstep). Jev with lookahead stays below `oracle-5s` (1,109), which scores the same rollouts with a fixed formula.

### 2026-09-30 (browser recording of a plain LLM)

- The browser accepts `?model=llm:<ollama-model>` and plays with the plain chat policy through the dev proxy.
- Recorded `docs/media/demo_qwen35_4b.mp4` (30 s, seed 100, `llm:qwen3.5:4b`): 1,370 points, 109 pellets, no lives lost, about 260 ms per decision.

### 2026-09-30 (latency breakdown on M3 Ultra and RTX 5090)

- Added `scripts/bench_prefill_decode.py`: times prefill and decode separately through Ollama or a llama.cpp server, with a unique tag per request so no prompt cache is reused.
- M3 Ultra, Ollama: `tev1:4b` prefill 178 ms for 383 tokens (0.47 ms per token), 12.0 ms per further output token; `pacman-0.8b` 41 ms and 5.5 ms.
- RTX 5090 (borrowed through the GPU lease and restored), CUDA llama.cpp: `tev1:4b` prefill 50 ms through `llama-server`, 25 ms in `llama-bench`, 4.5-4.8 ms per output token; `pacman-0.8b` 17 ms and 11 ms, 2.0-2.1 ms.
- A first run without the per-request tag reused prompt caches across rounds and reported 19 ms prefill for `tev1:4b`. Another early run measured 457 ms end to end while Ollama was swapping models. Both were discarded.

### 2026-10-01 (distilling the Qwen teacher into 0.8B)

- Replayed the original training players on their original seeds to recover the same states with full game state, and relabeled them with `teacher-peek5s` (hard labels; the teacher server rejects logprobs): 54,325 requests in about 65 minutes at concurrency 8.
- Trained three 0.8B students with `pacman-0.8b`'s recipe on one leased RTX 5090 (released 23:35, restored 01:46, readiness back to 3/3): A on features with teacher labels, B on `features-peek5s` with teacher labels, C on features with the oracle's best option as a hard label.
- B agreed with the teacher on 96.9% of held-out states, scored 1,387 pellets in lockstep (teacher 1,396) and 765 in real time, at 84 ms p50. A scored 425 over three realtime runs and 417 in lockstep; C 406; `pacman-0.8b` reran at 300 and 400 against its documented 456.
- JevBench generality: B 0.61, A 0.53, C 0.46.
- Realtime runs today shared the Mac with heavy unrelated CPU load. Same-model runs varied by up to about 150 pellets.

### 2026-10-01 (lookahead only labels data)

- Decided that lookahead may label data but never enters a player's input in a comparison: an exact engine rollout is close to seeing the future. Results with lookahead in the input (sections 13, 16, and student B in 18) stay as records, not player comparisons.
- The distilled 0.8B to cite is `pacman-0.8b-qwen`: teacher Qwen3.8-27B with lookahead, student on features only.

### 2026-09-30 (iPhone demo with three players)

- The iOS app now plays as one of three players, picked with a segmented control above the maze or with `decisionpacman://control?model=finetuned|phi4-mini|jev`: the fine-tuned 0.8B on device, phi4-mini on device, or Jev in the cloud. The choice is remembered across launches. Switching frees the previous on-device model before loading the next. The status heartbeat adds `model`, `modelName`, and the native `engine` state (player, model file, load state, load time).
- Model files are read from Documents: `pacman-0.8b-qwen.gguf` (falling back to `model.gguf`) for the fine-tuned player and `phi4-mini.gguf` for phi4-mini, overridable by name in `Documents/players.json`.
- phi4-mini on device (`ios/DecisionPacman/ChatMoveEngine.swift`) asks the way `llm:phi4-mini` asks Ollama: `teacherPrompt` with the JSON answer line, in phi4-mini's chat template as Ollama renders it, greedy decoding under a GBNF grammar equivalent to the move schema, probabilities from the logits at the move token renormalized over the options. Generation stops once the move is decided; the rest is forced by the grammar.
- Jev in the app goes through a native handler that posts the page's request body to the fixed TypeSafe endpoint with the key from `Documents/jev_key.txt`. The key never enters the page, the bundle, or the status file.
- Removed the Latency tab and `decisionpacman://bench`. The benchmark prompts moved to `ios/mac_check/bench_prompts.json`, which `scripts/bench_prefill_decode.py` now uses by default. `ios/scripts/mac_check.sh` runs either engine on the Mac. `ios/mac_check/chat_prompts.json` holds 40 real game prompts with Ollama's recorded `llm:phi4-mini` answers (`scripts/export_chat_prompts.ts`).
- Mac check, M3 Ultra, llama.cpp b11298 on Metal. Fine-tuned 0.8B (Q8_0): same choice as Ollama on 60 of 60, max probability difference 0.0003, p50 36 ms. Unchanged from the first iOS check.
- phi4-mini (Q4_K_M, the Ollama blob), 40 game states: 40 of 40 answers legal, 40 of 40 the same move as Ollama, probabilities equal to 4 decimals, prompt token counts equal to Ollama's on all 40. p50 168-173 ms, p90 about 200 ms: prefill p50 117 ms for about 290 tokens, 7 output tokens to reach the move, about 9.5 ms per decode step. The first answer takes about 370 ms because it builds the option-token tables.
- Not measured on the phone yet. A rough estimate from the 0.8B (380 ms on the phone against 36 ms here) and phi4-mini's size: 1.5-2 s per decision at first, and possibly twice that once the phone heats up. Expect a high stale rate at 1x; 0.5x speed may suit a phi4-mini demo better. Memory: 2.3 GB of weights, 544 MB of cache, and 403 MB of compute buffers on the Mac.

### 2026-10-01 (iPhone demo on the phone)

- The fine-tuned player now loads the distilled `pacman-0.8b-qwen` (falls back to `model.gguf`). All three players see the same current-state input.
- On the iPhone 16 Pro Max, 1x speed: the distilled 0.8B answered in about 400 ms per decision (p50 393-425 ms over the first 18 decisions); phi4-mini loaded in 4.9 s and answered in 2.2-2.9 s per decision, without memory trouble; Jev over Wi-Fi answered in about 125-155 ms with the key read from Documents.

### 2026-10-01 (quarter speed)

- Added a 0.25x speed option for the slow on-device phi4-mini. On the iPhone 16 Pro Max at 0.25x, phi4-mini still answered in 2.5-2.6 s per decision with the thermal state at "serious", and 8 of 18 decisions in the first minute arrived after Pac-Man had passed the junction.

### 2026-10-01 (wait for the model)

- Added a "Wait" switch (checkbox, `?wait=1`, `decisionpacman://control?wait=1`, `window.__pacman.setWaitForModel`). The model is asked about the very next junction while the game keeps moving; if Pac-Man is about to reach that junction before the answer arrives, the whole game (ghosts and timers included) holds until it does. A slow model is then judged on its choices, not its speed. The status report includes `waitForModel`.
- On the iPhone 16 Pro Max, phi4-mini at 1x with Wait on: 26 decisions, none late, about 2.2 s each; the game held in the windows where it waited (0-11 ticks per 2 s instead of 60). Without Wait at 0.25x, 8 of 18 decisions had been late.

### 2026-10-01 (documentation for readers)

- Rewrote README results first (the apples-to-apples ladder: features input, no lookahead, realtime; lockstep rows listed separately; clips; quickstart; a reproduce-each-result index).
- Added `docs/results.md` as the single place for current numbers and caveats.
- Added `docs/guides/` (run-models, evaluate, distill, iphone), including the 0.25x speed and the Wait switch.
- Updated the RFC summary and components to the current system and added a note to the PRD that it is the original requirements.
- `docs/model_evaluation.md` stays the lab notebook with unchanged section numbers, plus a pointer to `results.md`.
- Fixed stale paths and commands: `src/oracle/` and `src/ui/` in AGENTS.md (the oracle is `src/agent/oracle.ts`), and the headless command in `docs/test.md` (`--policy`, not `--model`). No code changes.

### 2026-10-01 (smaller chat models for the phone)

- Screened small plain chat models in lockstep (features input, seeds 100-109, 10 games) before stopping the search: `qwen3.5:0.8b` 35 pellets, `gemma3:1b` 45, `llama3.2:1b` 117, `qwen3:1.7b` 121, against Jev 161 in lockstep. None of the 1B-class models reached Jev.
- Tried qwen3.5 4B as a fourth iPhone player (Qwen3.5 chat template with thinking off, matched to Ollama's `think: false` rendering on 12 states). On the iPhone 16 Pro Max with Wait on it took 2.75-2.9 s per decision after a 3.4 s load, about the same as phi4-mini (3.0-3.2 s in the same session, thermal state `serious`). The app keeps three players: the distilled 0.8B, phi4-mini, and Jev.
- Ollama's `qwen3.5:4b` blob does not load in the app's llama.cpp (`qwen35.rope.dimension_sections has wrong array length; expected 4, got 3`); a standard GGUF such as unsloth's `Qwen3.5-4B-Q4_K_M.gguf` does.
### 2026-10-01 (published model, oracle to the FAQ)

- Published `pacman-0.8b-qwen` on Hugging Face (`grapeot/decision-pacman-0.8b-GGUF`): the Q8_0 GGUF and a Modelfile with the training system prompt. README, `results.md`, and the run, distill, and iPhone guides now install it from there.
- Reframed the public docs around off-the-shelf models against the 0.8B distilled from the LLM teacher. The rollout search `oracle-5s` and the student trained on it, `pacman-0.8b`, moved to an FAQ in the README and `results.md`; `model_evaluation.md` is unchanged.
- Recorded `docs/media/demo_pacman_08b_qwen.mp4` (browser, seed 100, 1x, 30 s, M3 Ultra under unrelated background load, load average about 20): score 3,320, 176 pellets, no lives lost. The first take opened on about 27 s of blank page while Vite optimized dependencies in a fresh checkout; the second take, with the cache warm, is the one kept.

### 2026-10-01 (Mac check reference for the distilled model)

- Replaced the reference probabilities in `ios/mac_check/bench_prompts.json` with those Ollama returns for `pacman-0.8b-qwen` (raw prompt, one output token, letter log-probabilities renormalized over the options). The same method reproduced the old `pacman-0.8b` references exactly (20 of 20 choices, largest difference 0.0).
- `ios/scripts/mac_check.sh decision` with the GGUF downloaded from Hugging Face: same choice on 60 of 60, largest probability difference 0.0004, 34 ms per decision.
- A screen recording on the iPhone 16 Pro Max showed the distilled model at a median of about 505 ms per decision with 25-28% late answers at 1x.

### 2026-10-01 (phi4-mini and iPhone clips)

- Recorded `docs/media/demo_phi4_mini.mp4` (browser, seed 100, 1x, 30 s, `llm:phi4-mini`, features, M3 Ultra under unrelated background load, load average about 17-20): score 1,710, 143 pellets, two lives lost, p50 about 150 ms per decision. A short throwaway take warmed Vite's dependency cache first, so the clip opens on the game.
- Added `docs/media/iphone_pacman_08b_qwen.mp4`, a 29-second screen recording of `pacman-0.8b-qwen` on the iPhone 16 Pro Max (1x, random seed, status bar cropped): about 505 ms per decision, 25-28% late answers, score 2,500, 174 pellets, no lives lost. README, `results.md`, and the iPhone guide link both clips.

### 2026-10-05 (hosted teacher)

- Added hosted-API support to the teacher: `TEACHER_API_KEY` (bearer), `TEACHER_EXTRA_BODY` (JSON merged into each request, for each API's thinking switch and provider pinning), and `TEACHER_MAX_RETRIES` (retries on 429/5xx, honoring Retry-After). `gen-data` records input, output, and reasoning tokens and any reported cost per request, and takes `--label-every k`. Added `scripts/compare_teacher_labels.py`, which joins two runs on player, seed, and tick and projects cost.
- DeepSeek V4.1 Flash on Ollama Cloud (`deepseek-v4.1-flash`, `reasoning_effort: none`), features + 5 s rollout facts, greedy seed 2000, 22 states, concurrency 1: 455.4 input and 31.3 output tokens per request, 0.60 s p50 and 0.78 s p90, 0 failures, 21 of 22 the same as the Qwen3.8-27B labels. At $0.15/$0.60 per million off-peak, $0.087 per 1,000 requests; 54,325 requests about $4.73 off-peak, $9.46 peak.
- Published prices for Qwen3.8-27B on 2026-10-05, applied to DeepSeek's measured token counts as an estimate, put the 54,325-request run between about $6.90 (DeepInfra) and about $27 (Groq, Cerebras).

### 2026-10-05 (training the student on a Mac with MLX)

- Added the Apple-silicon training path: `training/train_mlx.py` (same rows, prompt, letter-only loss, LoRA rank 16 / alpha 32, lr 1e-4, batch 32, row order as `train.py`), `merge_mlx.py`, `export_gguf_mlx.sh`, `score_mlx.py`, `test_train_mlx.py`, `requirements-mlx.txt` (`mlx` 0.32.3, `mlx-lm` 0.32.0), and `docs/guides/distill-mlx.md`. Results in section 20 of `model_evaluation.md`.
- M3 Ultra (80-core GPU, 512 GB), Mac otherwise idle: one epoch (987 steps) in 74.5 minutes of training loop (4.22 s per step, 2,807 prompt tokens per second, peak 47.8 GB at micro-batch 16), against 0.87 s per step on the RTX 5090. Final validation agreement 59.6% (5090 two-epoch run: 61.9%). A two-epoch run ran 1,600 steps at 4.13 s per step before the launching tool's time limit killed it; its validation agreement was flat at 59-60% from step 600.
- Served as Q8_0 in Ollama 0.35.1 (`pacman-0.8b-qwen-mlx`): same choice as the MLX weights on 49 of 50 states; agreement with the teacher on 1,000 held-out states 59.4% against 61.9% for `pacman-0.8b-qwen`, both 52 ms p50.
- In-game, seeds 100-109, features, M3 Ultra, Ollama 0.35.1, the two models alternated (load average 6-12): `pacman-0.8b-qwen-mlx` realtime 343, 413, 359 (mean 371, p50 53 ms), lockstep 352; `pacman-0.8b-qwen` realtime 366, 403, 511 (mean 427, p50 53 ms), lockstep 417 (same as section 18).
- Ollama 0.35.1 answers `/v1/systemone` with "does not support decision" for a model without `CAPABILITY decision`, including `pacman-0.8b-qwen` imported under 0.35.0. `/api/create` with `"capabilities": ["decision"]`, or a Modelfile with `CAPABILITY decision` (including `FROM <existing model>` plus that line), fixes it without touching the weights.

### 2026-10-08 (OpenAI Decisions API, GPT-6 Luna)

- Added the `openai:<model>` policy (`src/agent/decisions.ts`) for OpenAI's Decisions API (`POST /v1/decisions`, public beta since 2026-10-06; `gpt-6-luna` is the only model). The encoded state goes in as `input` and the move as one `choice` question with the encoder's instructions and the legal options; the reply's per-option probabilities are logged as for Jev. A refusal or an option outside the legal set fails the decision. `OPENAI_API_KEY` from the shell or `.env`; in the browser the dev server proxies `/openai` and adds the key. Tests: `tests/decisions.test.ts`; 91 passing.
- Smoke test, features, realtime, 1x, seeds 100-102, M3 Ultra, over the internet: 220, 223, 242 pellets, p50 160 ms, 5.8% stale, 0 failed of 1,047 decisions, 360 input tokens per decision.
- Two 10-game runs (seeds 100-109, 5-minute cap), each at the same time as a Jev run (load average about 6-8): GPT-6 Luna 238.9 and 234.6 pellets (p50 177 and 158 ms as mean of games, stale 6.6% and 4.7%, 0 failed or illegal of 6,640 decisions); Jev 187.8 and 177.5 (p50 115 and 113 ms, stale 3.9% and 2.7%, 1 HTTP 529). Paired by seed and run, +54.1 pellets per game for GPT-6 Luna, standard error 15.4, ahead in 18 of 20. Input 359 tokens per decision, 1.19 million per 10 games, $0.12 at $0.10 per million input tokens. Section 21 of `model_evaluation.md`.
- Browser, seed 100, 1x, through the dev-server proxy: p50 about 250 ms, p90 about 400 ms; 120 pellets and no lives lost in the first 30 s.

### 2026-10-10 (fork: random walk, Jev proxy, all-ghosts encoder)

- Added `random-<n>s` (`randomWalkPolicy`): one random exit per junction, and once every n game seconds a draw that turns back with probability 1/4. Plain `random` re-rolls "back" on every question and dithers. Headless, realtime, 1x, seeds 100-109, 5-minute cap, Apple M4 16 GB: random 27.6 pellets (the 30 in `results.md` predates later engine changes; the unmodified code also gives 27.6), random-1s 130.1, random-2s 144.6, greedy 200.7.
- The browser runs local Ollama models and hosted Jev side by side: `/decide` goes to Ollama, a new `/jev` proxy goes to TypeSafe with the key from the shell or `.env`, and `jev-latest` routes there. The model list always offers `jev-latest` and the random players.
- Local `pacman-0.8b-qwen` (Q8_0 from Hugging Face, Ollama 0.40.2) on an Apple M4 16 GB: p50 260-370 ms per decision, about 395 input tokens.
- Added the `features-4ghosts` encoder: each option lists every normal ghost within 15 steps, nearest first, as `[steps, coming]`, instead of only the nearest one. On 4,177 states from greedy and random play (seeds 5000-5039), 6.8% of options and 17.8% of decisions had two or more normal ghosts within 15 steps on one option. Listing them costs about 7 more input tokens per request on the local model.
- Jev (`jev-latest`), features against features-4ghosts, run at the same time, realtime, 1x, seeds 100-109, 5-minute cap, M4 over the internet: 202.7 against 162.9 pellets, 67 s against 48 s survived, p50 101 and 103 ms, stale 1.8% and 4.7%, 0 failed of 7,973 decisions. Paired by seed, features-4ghosts ate 39.8 fewer pellets per game (standard error 16.3) and was behind in 9 of 10. Jev chose "back" in 22.8% of decisions with the list against 9.3% with the single ghost. Input 609 and 619 tokens per decision; 2.81 and 2.08 million tokens, about $0.12 and $0.09 at the $0.042 per million seen on Vercel AI Gateway. One run each, so within run-to-run spread for Jev but consistent across seeds.
- Added the `features-ghosts` encoder: everything in `features`, plus `ghost2`-`ghost4` (with `_coming`) per option, every ghost's name, state, and shortest steps from Pac-Man, `ghost_mode` and `ghost_mode_seconds_left`, and the chase rules and reversal rules in the instructions. The `features` output is unchanged (identical on 30,911 states). About 985 input tokens per Jev request against 606.
- Jev, features against features-ghosts, run at the same time, realtime, 1x, seeds 100-109, 5-minute cap, M4: 189.4 against 157.5 pellets, 64 s against 65 s survived, p50 104 ms both, stale 1.6% and 4.5%, 0 failed of 8,739 decisions. Paired by seed, features-ghosts ate 31.9 fewer pellets per game (standard error 21.9), ahead in 4 of 10. Jev chose "back" in 17.9% of decisions against 6.6%. 2.63 and 4.34 million input tokens, about $0.11 and $0.18 at $0.042 per million. Both richer ghost encodings raised Jev's reversal rate two- to threefold and did not raise its score.
- Removed both experimental encoders (`features-4ghosts`, `features-ghosts`) afterwards; the numbers above are kept as a record. More ghost facts alone did not help Jev, and the jump in "back" answers when it sees more ghosts is a separate problem to study first.
- Why Jev turns back (the two features runs above, 8,935 decisions with a back option, 719 backs): 77% with no normal ghost within 15 steps on any option (of those, 54% had more pellets within 6 steps behind, 8% a closer pellet behind, 4% a closer power pellet, 35% no visible reason); 11% toward a ghost closer than any ahead (58 of them coming, 27 within 5 steps); 9% away from a ghost ahead only. 45% of backs were near-ties (top two within 0.15). 29% of applied backs were followed by another applied back within 1 s. A life was lost within 2 s after 2.4% of back answers against 0.5% of forward answers. In the 4-ghost and detailed runs the extra backs came almost only from states with no ghost within 15 steps (20.4% and 15.7% of all decisions, against 6.2%).
- Same-state test, 150 states with no normal ghost within 15 steps (greedy and random play, seeds 7000+, after the first 5 s), one Jev call per state and format: features 8.7% back (mean P(back) 0.19), 4-ghost list 20.7% (0.23), detailed 10.0% (0.24), features state with the detailed instructions 8.0% (0.20). The list format itself (an empty `ghosts: []` where features says `ghost: null`) doubles Jev's backs in calm states; the detailed format raises them only slightly on a fixed state, so its in-game rise (6.2% to 15.7%) is mostly amplification by repeated reversals. The longer instructions alone change nothing. At the start position Jev turned back on 150 of 150 first decisions (back shows 10 pellets within 6 steps against 5-6 ahead).
- Added the back lock (`ReverseLock` in `src/agent/loop.ts`, `withoutBack` in `src/encoders/types.ts`): for a set number of game seconds after Pac-Man's next junction changes (it turned around or passed a junction; corners do not count), "back" is left out of the options, keys, and criteria. Used by the browser loop (`?lock=<s>`) and the headless runner (`--reverse-lock <s>`); off by default, so ladder numbers are unchanged. Tests: `tests/reverse_lock.test.ts`; 102 passing.
- Jev, features, lock off against lock 1 s, run at the same time, realtime, 1x, seeds 100-109, 5-minute cap, M4: 177.2 against 205.0 pellets, 54 s against 66 s survived, p50 98 ms both, stale 2.0% and 0.6%, 0 failed of 8,532 decisions. Paired by seed, +27.8 pellets per game with the lock (standard error 7.8), ahead in 10 of 10. With the lock, back was offered in 634 of 4,718 decisions (13%) and chosen in 49 (1.0% of all decisions, against 8.9% without); applied backs followed by another back within 1 s fell from 86 to 1, and backs toward a closer ghost from 37 to 8. 2.32 and 2.64 million input tokens, about $0.10 and $0.11 at $0.042 per million. The lock is a rule of the game interface, applied the same way to every player, so lock-on scores belong in a separate column from the ladder.
- Added the `features-ghosts` encoder for training students: everything in `features`, plus `ghost2`-`ghost4` (with `_coming`) per option, and for every ghost its name, state, heading (`dir`), and shortest steps from Pac-Man, with `ghost_mode` and `ghost_mode_seconds_left`. Its extra instructions are one short paragraph, since a trained student learns the fields from data. `features` is unchanged (identical on 30,911 states). On local `pacman-0.8b-qwen`, M4: 600 input tokens against 402, p50 532 ms against 465 ms (same 100 states, alternated).
- Added `oracle-pellets-5s`: the same rollouts as `oracle-5s`, valued at 10 per pellet eaten instead of game points (no credit for frightened ghosts or the power-pellet bonus); the death penalty stays. `oracle-5s` is unchanged. Realtime, 1x, seeds 100-109, 5-minute cap, M4, run at the same time: oracle-5s 1,115.6 pellets (23 frightened ghosts eaten per game, 6 deaths in 10 games), oracle-pellets-5s 1,171.1 (10 ghosts, 1 death); +55.5 per game, standard error 43.5, ahead in 8 of 10.
- Started dataset `p1` (gitignored `data/p1`): states from oracle-pellets-5s, greedy, and random play on the original seed ranges (train 1000-1019, 2000-2029, 3000-3029; validation 900-902, 920-924, 940-944), all labeled by oracle-pellets-5s with per-option values, each state stored in `features` and, under `alt`, `features-ghosts`. Two students can be trained from the same states and labels, differing only in input. 64,849 states (56,645 train, 8,204 validation) in about 2.5 minutes of wall time on the M4 (about 185 labels per second per process, six processes). `build_sft.py` with soft targets (temperature 50): `sft_features` 30,566 train and 4,850 validation rows, `sft_ghosts` 35,711 and 6,414. The ghost encoding dedups less, because states that look the same in `features` can differ in the ghosts further away, so the two sets differ in size. Longest `sft_ghosts` prompt about 3,000 characters, under the 1,024-token limit. Added `training/kaggle_train.ipynb`: a 20-step trial on a T4 (Gated DeltaNet on compute capability 7.5, fp16 loss, seconds per step), then both students in parallel, one per GPU.

## Lessons Learned
- Run hosted players you compare at the same time, not one after the other. Hosted latency moves with the provider's load: in one session Jev's p90 reached 832 ms in one game against about 150 ms in most, and one GPT-6 Luna game ran at twice its usual p50. Running both in parallel puts them under the same conditions, and pairing by seed then removes most of the game-to-game spread.
- Hosted APIs ignore `chat_template_kwargs`. A request that turns thinking off on vLLM leaves it on at OpenRouter, Vercel AI Gateway, or Ollama Cloud, where it eats the 200-token answer budget. Send each API's own switch and check that reasoning tokens are 0.

- MLX is not torch with another backend. Check optimizer defaults (MLX's AdamW has `bias_correction=False` and `weight_decay=0.01`), cap the buffer cache when shapes vary between steps, and check gradients of new code against a reference, not just outputs: in `mlx` 0.32.3 a step-2 slice over an axis of length 2 has a wrong gradient and a right forward value.
- Launch runs of more than an hour detached from the agent tool that starts them, and checkpoint them: a run stopped by a tool's time limit after 1,600 of 1,974 steps left no adapter.
- Latency scales with input tokens (~0.8 ms per token for nimble on M3 Ultra). Putting the static maze first and the dynamic part last saved only ~35 ms, so prefix caching does not make large states cheap. Exact repeats of a request return in ~35–60 ms, so benchmarks must use fresh states or they will look far faster than a real game.
- On the full ASCII board, nimble's confidence stayed between 0.00 and 0.09. On per-direction facts it chose the obvious move with probability ~0.99. The representation matters more than the model size between nimble and tev1:4b.
- tev1:0.8b is fast but near chance on simple situations. Do not make it the default.
- The models fled frightened ghosts. Game rules have to be stated in `instructions`, because the model does not bring Pac-Man knowledge into the decision.
- Ollama's launch blog includes its own nimble Pac-Man example (91 ms per decision on an M5 Max, only legal moves as options). It is a recorded, turn-based replay (JSON in the page), with no code to reuse.
- The synthetic scenarios have one clearly correct move each. They test format and basic judgment, not play on real, ambiguous game states. Teacher and student quality claims need states sampled from the engine.
- The teacher server rejects `response_format: json_schema` (no constrained decoding), so labels come from parsing plain text. A prompt that asks for JSON was 100% parseable in 120 calls.
- Keep "turn around now" separate from turns. When a junction exit points opposite to the current heading (after a corner), treating it as a reversal made Pac-Man dither in place. `StepInput.reverse` turns around, and `StepInput.turn` plans a turn for one specific junction tile.
- Features must count the corridor between Pac-Man and the junction. Measuring only from the junction outward hid pellets and ghosts on the way and made "back" look closer than it was.
- Latency has to become lookahead. Asking about a junction Pac-Man reaches before the answer returns wastes the call. The agent now asks about the junction after next once the near turn is planned and Pac-Man is within the latency distance. Stale answers fell from 22% to 10%, and pellets rose from 138 to 169.
- In the browser, wait one tick after queuing an answer before asking again. Otherwise the next request is planned from a state without the new turn. This alone took the browser stale rate from about 40% to 13%.
- Do not ask during frozen phases (ready, dying). Identical states hit the server's cache and pull the latency p50 down to about 40 ms, which misrepresents the real 200 ms.
- The greedy baseline scored 238 pellets without lookahead and 201 with it, because it decides earlier on older information. Compare baselines only within the same agent setup.
- Many teacher states are ties: no ghost nearby and equal food on several exits. A hard label on a tie is noise for the student. Before training, either down-weight ties, keep the teacher's reason to filter them, or ask for a distribution instead of a single move.
- Sequential lockstep labeling runs at about 3 states per second. 20,000 states would take about 2 hours on one game. Several games in parallel against the shared teacher endpoint cut that, within a concurrency cap.
- The engine is a perfect simulator, so rollout search beats any model we tried by a wide margin, and it runs locally in milliseconds. Label training data with the oracle, not an LLM. Keep the LLM as a comparison point.
- 30-second windows hide the difference between players. They are mostly pellet collection early in the level. Survival only separates over minutes.
- zsh does not word-split `$var`. `set -- $cfg` in a loop passed both arguments as one. Use functions with explicit arguments.
- Match Ollama's scoring prompt exactly when fine-tuning for `/v1/systemone`. It is a JSON user message (`context`, `schema` with lettered choices, then `Requested field`) under the model's system prompt, with thinking off, scored on the option letters only. Training on the same softmax over letters makes the served model behave like the trained one (0.010 median probability difference).
- Project only the answer position through the output layer. Qwen3.5's 248k-token vocabulary makes full logits several GB per batch.
- Ollama 0.35.0 rejects `CAPABILITY` in a Modelfile but serves `/v1/systemone` for an imported GGUF anyway. Create the model through `/api/create` to set the system prompt exactly, including its leading and trailing newlines.
- llama.cpp's converter requirements pin transformers 4.57.6, but Qwen3.5 tokenizers need transformers 5.5 or later. Install the requirements, then upgrade transformers.
- Measure a reported stall before fixing it. Per-window frame counts, the largest frame gap, and dropped simulated time, compared between AI and human mode and split by game phase, showed that on-device Metal inference does not starve the web view's frame loop. The planned fixes (smaller llama.cpp batches, gaps between decisions, a different catch-up rule) would have changed behavior for a problem that did not exist.
- Launch the app with `devicectl device process launch --activate` when measuring. Without it, one launch left the app behind other UI: `requestAnimationFrame` stopped at once and timers stopped a few seconds later, which looks exactly like a frozen game. Check that the heartbeat's frame count is non-zero before reading anything else.
- Ollama's chat logprobs are the model's raw distribution, before the JSON schema masks tokens: the first token's alternatives include a code fence. Renormalize over the legal options at the move token. On Ollama 0.35.0, `gemma4:e4b` returns logprobs only for the first generated token, so it gives no option probabilities.
- On the phone, decision latency is not a constant. Sustained back-to-back inference nearly doubles it within a minute, so lookahead sizing and stale-rate numbers from the first 30 s of a game are optimistic.
- phi4-mini's LongRoPE has two sets of rope factors, and llama.cpp picks the long ones whenever the context is larger than the model's original 4096 tokens, regardless of the prompt's length. Ollama on this Mac evidently runs with a larger context (its answers match only the long factors), so the evaluated `llm:phi4-mini` used them. With a 2048-token context the same engine and prompts agreed with Ollama on 33 of 40 moves (median max probability difference 0.09); with 4352 or 8192 it agreed on 40 of 40. `yarn_orig_ctx` does not change the choice in b11298. Match the context regime before comparing a phi-family model across runtimes.
- On Metal, `llama_decode` returns before the GPU finishes. Call `llama_synchronize` before reading a prefill time, or prefill looks like 4 ms and decode absorbs it.
