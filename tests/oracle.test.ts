import { describe, expect, it } from "vitest";
import { makePolicy } from "../src/agent/factory.ts";
import { nearestFood, oraclePolicy, peek, rollout, rolloutValue } from "../src/agent/oracle.ts";
import { createGame, step } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { ENCODERS } from "../src/encoders/index.ts";
import type { GameState } from "../src/engine/types.ts";

function playing(seed = 1): GameState {
  const s = createGame(seed);
  while (s.phase !== "playing") step(s);
  return s;
}

describe("oracle", () => {
  it("does not change the live game while searching", async () => {
    const s = playing();
    const before = JSON.stringify({ ...s, food: [...s.food] });
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    await oraclePolicy({ horizonSeconds: 2 }).decide(enc, undefined, s);
    expect(JSON.stringify({ ...s, food: [...s.food] })).toBe(before);
  });

  it("is deterministic for the same state", () => {
    const s = playing();
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    for (const key of enc.keys) expect(rolloutValue(s, enc, key, 60, 0)).toBe(rolloutValue(s, enc, key, 60, 0));
  });

  it("turns back from a ghost waiting at the junction ahead", async () => {
    const s = playing();
    for (const g of s.ghosts) Object.assign(g, { state: "house", x: 13.5, y: 14 });
    s.dotsEatenThisLife = -1000;
    s.ticksSinceDot = -100000;
    Object.assign(s.pac, { x: 9, y: 5, dir: "left", lastDir: "left" });
    // Blinky sits on the next junction (6,5), heading toward Pac-Man.
    Object.assign(s.ghosts[0], { x: 6, y: 5, dir: "right", state: "active", frightened: false });
    const dp = findDecisionPoint(s);
    const enc = ENCODERS.features.encode(s, dp);
    const d = await oraclePolicy({ horizonSeconds: 3 }).decide(enc, undefined, s);
    expect(d.choice).toBe("back");
  });
});

describe("ascii-full encoder", () => {
  it("draws all 31 rows with a column header and marks Pac-Man", () => {
    const s = playing();
    const enc = ENCODERS["ascii-full"].encode(s, findDecisionPoint(s));
    const map = (enc.state as string).split("Map:\n")[1].split("\n");
    expect(map).toHaveLength(32);
    expect(map.slice(1).some((row) => row.includes("P"))).toBe(true);
    expect(enc.state as string).toContain("Pac-Man at column");
  });

  it("peek reports a simulated future as plain facts", () => {
    const s = playing();
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    for (const key of enc.keys) {
      const o = peek(s, enc, key, 90);
      expect(o.dies_after_s).toBeNull();
      expect(o.pellets).toBeGreaterThan(0);
      expect(o.points).toBe(o.pellets * 10 + o.power_pellets * 50);
    }
  });
});

describe("pellet oracle", () => {
  it("values an option by pellets eaten, not by points", () => {
    const s = playing();
    for (let i = 0; i < 120; i++) step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    for (const key of enc.keys) {
      const end = rollout(s, enc, key, 150, 0);
      if (end.stats.deaths > s.stats.deaths) continue;
      const eaten = end.stats.pelletsEaten - s.stats.pelletsEaten;
      expect(rolloutValue(s, enc, key, 150, 0, "pellets")).toBe(10 * eaten - nearestFood(end));
    }
  });

  it("differs from the score objective only by points that are not 10 per pellet", () => {
    const s = playing();
    for (let i = 0; i < 120; i++) step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    for (const key of enc.keys) {
      const end = rollout(s, enc, key, 150, 0);
      if (end.stats.deaths > s.stats.deaths) continue;
      const otherPoints = end.score - s.score - 10 * (end.stats.pelletsEaten - s.stats.pelletsEaten);
      expect(rolloutValue(s, enc, key, 150, 0, "score") - rolloutValue(s, enc, key, 150, 0, "pellets")).toBe(otherPoints);
    }
  });

  it("is built by name, searches without a model, and keeps oracle-5s unchanged", async () => {
    const s = playing();
    for (let i = 0; i < 120; i++) step(s);
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    for (const name of ["oracle-pellets-5s", "oracle-pellets-3s", "oracle-5s"]) {
      const p = makePolicy(name, 1, { decisionBaseUrl: "http://127.0.0.1:9" });
      expect(p.name).toBe(name);
      const d = await p.decide(enc, undefined, s);
      expect(Object.keys(d.values!).sort()).toEqual([...enc.keys].sort());
    }
  });
});
