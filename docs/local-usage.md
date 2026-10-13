# Local usage

This guide is for the `local/learning-poc` branch. It covers how to start the game, how to let hosted Jev play, and how to let the local model play.

## 1. Setup (once)

1. Install dependencies:
   ```bash
   npm install
   ```
2. Install and start Ollama (0.35 or later):
   ```bash
   brew install ollama
   brew services start ollama      # starts at login; stop with: brew services stop ollama
   ```
3. Download the distilled model and import it into Ollama:
   ```bash
   hf download grapeot/decision-pacman-0.8b-GGUF pacman-0.8b-qwen-Q8_0.gguf Modelfile.pacman-0.8b-qwen --local-dir ./tmp-model
   cd tmp-model
   echo 'CAPABILITY decision' >> Modelfile.pacman-0.8b-qwen
   ollama create pacman-0.8b-qwen -f Modelfile.pacman-0.8b-qwen
   cd .. && rm -rf tmp-model      # Ollama keeps its own copy, so the download can go
   ```
   If `hf download` stalls, download the same file with `curl -L -C -`. The `-C -` flag resumes where the last attempt stopped.
4. Create `.env` in the project root. Git ignores it, so the key is never committed:
   ```bash
   VITE_DECISION_BASE_URL=http://localhost:11434
   VITE_DEFAULT_MODEL=pacman-0.8b-qwen
   TYPESAFE_API_KEY=<your TypeSafe key>
   ```

## 2. Start the game

```bash
npm run dev
```

Then open http://localhost:5173/ in a browser.

The dev server reads the key from `.env`. The key is used only inside the dev server and never reaches the page.

## 3. Pick a player in the browser

In the right panel, set **Player** to `Model`. Then pick a player in the **Model** box:

| Model name | Who plays | Where requests go |
|---|---|---|
| `pacman-0.8b-qwen` | the distilled 0.8B model (default) | `/decide` → local Ollama |
| `jev-latest` | TypeSafe's hosted Jev | `/jev` → `https://api.typesafe.ai` |
| `random` | a random option every time it is asked (dithers in place) | no request |
| `random-1s` | a random exit per junction; once per second, go on or turn back at 3:1 | no request |
| `random-2s` | the same, once every 2 seconds | no request |

**To open the suggestion list:** clear the text in the Model box first, then click the box. Chrome only shows suggestions that match the text already there. Press Enter after picking one.

**Leave the Endpoint box at `/decide`.** The page sends `jev-latest` to `/jev` on its own.

With Player set to `Keyboard`, you play with the arrow keys or WASD. Space pauses, R restarts, M mutes.

### Picking a player by URL

- Local model: http://localhost:5173/?model=pacman-0.8b-qwen
- Jev: http://localhost:5173/?model=jev-latest
- Random walk: http://localhost:5173/?model=random-1s
- Play yourself: http://localhost:5173/?mode=human

More URL parameters:
- `&wait=1`: the game holds at each junction until the model answers, so a slow model is judged on its choices, not its speed.
- `&speed=2`: the game runs at 2x speed.
- `&seed=100`: fixes the game's random seed, so every start is the same.
- `&sound=0`: turns the sound off.
- `&lock=1`: the back lock. For 1 game second after Pac-Man turns around or passes a junction, "back" is not offered, so a model without memory cannot flip back and forth. The HUD shows `back lock 1s` while it is on. Off by default.

## 4. Headless games on the command line

The command line reads `.env` on its own. Each run writes its logs under `runs/`.

```bash
# Local model
npm run headless -- --policy pacman-0.8b-qwen --games 1 --seed 100 --max-seconds 60

# Jev (the command-line name is jev, not jev-latest)
npm run headless -- --policy jev --games 1 --seed 100 --max-seconds 60

# Random walk and the hand-written rule
npm run headless -- --policy random-1s --games 1 --seed 100 --max-seconds 60
npm run headless -- --policy greedy --games 1 --seed 100 --max-seconds 60
```

Two more names for experiments:
- `--policy oracle-pellets-5s`: the rollout search valued by pellets eaten instead of points. It sees 5 seconds of the future, so it is a labeler and a reference, not a ladder player.
- `--encoder features-ghosts`: `features` plus every ghost (`ghost2`-`ghost4` per option, and each ghost's name, state, heading, and steps), with the scatter/chase mode and its timer. It is meant for students trained on it; Jev read the earlier ghost-list formats poorly.

To train students on these encodings with oracle labels on a GPU server, see [guides/distill-oracle-gpu.md](guides/distill-oracle-gpu.md).

Add `--reverse-lock 1` to any of these to turn on the back lock (the same as `&lock=1` in the browser). It is off by default, so ladder numbers stay comparable.

### The same comparison as the README ladder

The README ladder uses seeds 100-109, 10 games, and at most 5 minutes per game:

```bash
npm run headless -- --policy random-1s --games 10 --seed 100 --max-seconds 300
```

The last line of output is a JSON summary. `meanPellets` is the mean number of pellets eaten.

## 5. Numbers measured on this machine

Machine: Apple M4, 16 GB memory. Seeds 100-109, 10 games, at most 5 minutes per game.

| Player | Mean pellets eaten |
|---|---|
| `random` | 27.6 |
| `random-1s` | 130.1 |
| `random-2s` | 144.6 |
| `greedy` | 200.7 |

On this machine the local `pacman-0.8b-qwen` takes about 260-370 ms per decision. The 55 ms in the README was measured on a faster machine.

## 6. Notes

- **The browser's PELLETS counter is pellets left, not pellets eaten.** It starts at 244, counts down, and resets to 244 on each new level. The ladder and the command line report pellets eaten, which keeps adding up across levels.
- **Jev costs money.** It is a hosted service, and the game sends about 4-5 requests per second while it plays. When you are not watching, switch Player to `Keyboard` or close the page.
- **Keep the key in `.env` only.** Never put it in a file that gets committed.
- **If the local model does not answer:** run `curl localhost:11434/api/version` to check that Ollama is running. If it is not, run `brew services start ollama`.
- **To watch `tev1:4b`:** download it first with `ollama pull tev1:4b` (several GB), then type `tev1:4b` in the Model box.
