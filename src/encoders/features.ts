import { distancesThrough, type DecisionPoint } from "../engine/decision.ts";
import { TPS } from "../engine/game.ts";
import { COLS, Food, ROWS, index, wrapX } from "../engine/maze.ts";
import { VEC, type Dir, type GameState, type Ghost } from "../engine/types.ts";
import { optionKeys, type EncodedDecision, type Encoder, type OptionKey } from "./types.ts";

const GHOST_HORIZON = 15;
const PELLET_WINDOW = 6;
const FOOD_HORIZON = 40;
const POWER_HORIZON = 25;

const INSTRUCTIONS =
  "You control Pac-Man in a maze. Choose which way to go at the next junction. " +
  "Eating pellets scores points. Touching a normal ghost loses a life. " +
  "After a power pellet, ghosts are frightened for a few seconds and can be eaten for bonus points. " +
  "Each option lists facts about the path through that exit: ghost = steps to the nearest normal ghost " +
  "(null if none within 15), ghost_coming = that ghost is moving toward you, edible = steps to the nearest " +
  "frightened ghost, pellets = pellets within 6 steps, food = steps to the nearest pellet, power = steps to " +
  "the nearest power pellet. Steps count from Pac-Man, including the corridor up to the junction. " +
  "'back' means turn around right now instead of going to the junction.";

// features-ghosts adds every ghost. A trained student learns the fields from data, so the extra text stays short
// (every instruction token is prefill latency).
const INSTRUCTIONS_GHOSTS =
  INSTRUCTIONS +
  " ghost2, ghost3, ghost4 (each with _coming) = the next nearest normal ghosts within 15 steps on that path. " +
  "ghosts = every ghost: name, state (normal, frightened, in_house, returning = eyes, harmless), dir = its heading, " +
  "steps = shortest path from Pac-Man (null if unreachable). ghost_mode = scatter or chase; all ghosts turn around " +
  "when ghost_mode_seconds_left reaches 0 (null = never).";

interface OptionFacts {
  ghost: number | null;
  ghost_coming?: boolean;
  ghost2?: number;
  ghost2_coming?: boolean;
  ghost3?: number;
  ghost3_coming?: boolean;
  ghost4?: number;
  ghost4_coming?: boolean;
  edible?: number | null;
  pellets: number;
  food: number | null;
  power?: number | null;
}

function isDangerous(g: Ghost): boolean {
  return g.state === "active" && !g.frightened;
}

function isEdible(g: Ghost): boolean {
  return g.state === "active" && g.frightened;
}

function ghostState(g: Ghost): string {
  if (g.state === "house" || g.state === "leaving") return "in_house";
  if (g.state === "eaten" || g.state === "entering") return "returning";
  return g.frightened ? "frightened" : "normal";
}

function ghostTile(g: Ghost): { x: number; y: number } {
  return { x: wrapX(Math.round(g.x)), y: Math.round(g.y) };
}

interface PathTile {
  x: number;
  y: number;
  /** Steps from Pac-Man to this tile center. */
  steps: number;
}

/**
 * Facts for one option. `dist` holds BFS distances through the option's exit,
 * `offset` converts them to steps from Pac-Man, and `prefix` lists the tiles
 * Pac-Man passes before reaching that exit (the corridor to the junction).
 */
function factsFor(
  s: GameState,
  dist: Int16Array,
  offset: number,
  prefix: PathTile[],
  allGhosts = false,
  /** Filled with each ghost's steps through this option (any state, no horizon), by index in s.ghosts. */
  reach: (number | null)[] = [],
): OptionFacts {
  const at = (x: number, y: number) => (y < 0 || y >= ROWS ? -1 : dist[index(x, y)]);

  let ghost: number | null = null;
  let coming = false;
  let edible: number | null = null;
  const near: [number, boolean][] = [];
  const consider = (g: Ghost, total: number, isComing: boolean) => {
    const i = s.ghosts.indexOf(g);
    if (reach[i] == null || total < reach[i]!) reach[i] = total;
    if (isDangerous(g) && total <= GHOST_HORIZON) near.push([total, isComing]);
    if (isDangerous(g) && total <= GHOST_HORIZON && (ghost === null || total < ghost)) {
      ghost = total;
      coming = isComing;
    }
    if (isEdible(g) && total <= GHOST_HORIZON && (edible === null || total < edible)) edible = total;
  };

  for (const g of s.ghosts) {
    const t = ghostTile(g);
    const onPrefix = prefix.findIndex((p) => p.x === t.x && p.y === t.y);
    if (onPrefix >= 0) {
      // On the corridor ahead: coming if its next tile is closer to Pac-Man.
      const v = VEC[g.dir];
      const prev = onPrefix > 0 ? prefix[onPrefix - 1] : null;
      const isComing = prev ? wrapX(t.x + v.x) === prev.x && t.y + v.y === prev.y : true;
      consider(g, Math.max(0, prefix[onPrefix].steps), isComing);
      continue;
    }
    const d = at(t.x, t.y);
    if (d < 0) continue;
    const v = VEC[g.dir];
    const ahead = at(wrapX(t.x + v.x), t.y + v.y);
    consider(g, Math.max(0, d + offset), ahead >= 0 && ahead < d);
  }

  let pellets = 0;
  let food: number | null = null;
  let power: number | null = null;
  const addFood = (f: number, total: number) => {
    if (total <= PELLET_WINDOW) pellets++;
    if (total <= FOOD_HORIZON && (food === null || total < food)) food = total;
    if (f === Food.Power && total <= POWER_HORIZON && (power === null || total < power)) power = total;
  };
  for (const p of prefix) {
    const f = s.food[index(p.x, p.y)];
    if (f !== Food.None) addFood(f, p.steps);
  }
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const i = index(x, y);
      const f = s.food[i];
      const d = dist[i];
      if (f === Food.None || d <= 0) continue;
      addFood(f, d + offset);
    }
  }

  const round = (v: number | null) => (v === null ? null : Math.round(v * 10) / 10);
  const facts: OptionFacts = { ghost: round(ghost), pellets, food: round(food) };
  if (ghost !== null) facts.ghost_coming = coming;
  if (allGhosts) {
    // near[0] is the nearest ghost already reported; a stable sort keeps the engine's ghost order on ties.
    near.sort((a, b) => a[0] - b[0]);
    const slots = [["ghost2", "ghost2_coming"], ["ghost3", "ghost3_coming"], ["ghost4", "ghost4_coming"]] as const;
    near.slice(1, 4).forEach(([d, c], k) => {
      facts[slots[k][0]] = round(d)!;
      facts[slots[k][1]] = c;
    });
  }
  if (s.frightTicks > 0) facts.edible = round(edible);
  if (power !== null) facts.power = round(power);
  return facts;
}

function makeFeaturesEncoder(name: string, allGhosts: boolean): Encoder {
  return {
    name,
    encode(s: GameState, dp: DecisionPoint): EncodedDecision {
      const keys = optionKeys(dp);
      const firstSteps = dp.distance - (dp.path.length - 1);
      const corridor: PathTile[] = dp.path.map((t, i) => ({ x: t.x, y: t.y, steps: firstSteps + i }));
      const options: Partial<Record<OptionKey, OptionFacts>> = {};
      const reach: (number | null)[] = s.ghosts.map(() => null);
      for (const key of keys) {
        if (key === "back") {
          // Turn around now: distances through the tile behind Pac-Man.
          const first = dp.path[0];
          options.back = factsFor(s, distancesThrough(first.x, first.y, dp.back as Dir), -firstSteps, [], allGhosts, reach);
        } else {
          options[key] = factsFor(s, distancesThrough(dp.x, dp.y, key), dp.distance, corridor, allGhosts, reach);
        }
      }

      const state: Record<string, unknown> = {
        lives: s.lives,
        pellets_left: s.foodLeft,
        to_junction: Math.round(dp.distance * 10) / 10,
      };
      if (s.frightTicks > 0) state.frightened_seconds_left = Math.round((s.frightTicks / TPS) * 10) / 10;
      if (allGhosts) {
        state.ghost_mode = s.mode;
        state.ghost_mode_seconds_left = Number.isFinite(s.modeTicks) ? Math.round((s.modeTicks / TPS) * 10) / 10 : null;
        state.ghosts = s.ghosts.map((g, i) => ({
          name: g.name,
          state: ghostState(g),
          dir: g.dir,
          steps: reach[i] === null ? null : Math.round(reach[i]! * 10) / 10,
        }));
      }
      state.options = options;

      const criteria: Record<string, string | null> = {};
      for (const key of keys) criteria[key] = key === "back" ? "turn around now" : null;
      return { encoder: name, state, instructions: allGhosts ? INSTRUCTIONS_GHOSTS : INSTRUCTIONS, criteria, keys, decision: dp };
    },
  };
}

export const featuresEncoder: Encoder = makeFeaturesEncoder("features", false);

/**
 * Features plus every ghost the engine knows about without looking ahead: ghost2-ghost4 per option, each ghost's
 * name, state, heading, and steps from Pac-Man, and the scatter/chase mode with the time until all ghosts reverse.
 */
export const featuresGhostsEncoder: Encoder = makeFeaturesEncoder("features-ghosts", true);
