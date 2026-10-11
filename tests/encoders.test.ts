import { describe, expect, it } from "vitest";
import { createGame, step } from "../src/engine/game.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { ENCODERS, optionToInput } from "../src/encoders/index.ts";
import { greedyPolicy } from "../src/agent/policies.ts";
import type { GameState } from "../src/engine/types.ts";

function playing(): GameState {
  const s = createGame(1);
  while (s.phase !== "playing") step(s);
  return s;
}

describe("features encoder", () => {
  it("offers exactly the junction exits plus back", () => {
    const s = playing();
    const dp = findDecisionPoint(s);
    const enc = ENCODERS.features.encode(s, dp);
    expect(enc.keys.sort()).toEqual(["back", "left", "up"]);
    expect(Object.keys(enc.criteria).sort()).toEqual(["back", "left", "up"]);
    const opts = (enc.state as { options: Record<string, unknown> }).options;
    expect(Object.keys(opts).sort()).toEqual(["back", "left", "up"]);
  });

  it("reports facts and never verdicts", () => {
    const s = playing();
    const text = JSON.stringify(ENCODERS.features.encode(s, findDecisionPoint(s)).state);
    expect(text).not.toMatch(/safe|danger|recommend|best|score/i);
  });

  it("stays within the token budget (character proxy)", () => {
    const s = playing();
    for (let i = 0; i < 600; i++) step(s);
    const text = JSON.stringify(ENCODERS.features.encode(s, findDecisionPoint(s)).state);
    expect(text.length).toBeLessThan(700);
  });

  it("sees a ghost that is close through one exit", () => {
    const s = playing();
    const dp = findDecisionPoint(s); // junction (12,23), exits left and up
    Object.assign(s.ghosts[0], { x: 12, y: 21, state: "active", frightened: false, dir: "down" });
    const enc = ENCODERS.features.encode(s, dp);
    const opts = (enc.state as { options: Record<string, { ghost: number | null; ghost_coming?: boolean }> }).options;
    expect(opts.up.ghost).not.toBeNull();
    expect(opts.up.ghost!).toBeLessThan(5);
    expect(opts.up.ghost_coming).toBe(true);
  });

  it("maps back to an immediate reverse and exits to turns planned at the junction", () => {
    const s = playing();
    const dp = findDecisionPoint(s);
    expect(optionToInput("back", dp)).toEqual({ reverse: true });
    expect(optionToInput("up", dp)).toEqual({ turn: { x: 12, y: 23, dir: "up" } });
  });

  it("feeds the greedy baseline", async () => {
    const s = playing();
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const d = await greedyPolicy().decide(enc);
    expect(enc.keys).toContain(d.choice);
  });
});

describe("ascii window encoder", () => {
  it("renders an 11x11 view with Pac-Man at the center", () => {
    const s = playing();
    const enc = ENCODERS["ascii-window"].encode(s, findDecisionPoint(s));
    const rows = (enc.state as string).split("\n").slice(1);
    expect(rows).toHaveLength(11);
    expect(rows[5][5]).toBe("P");
  });
});

describe("features-ghosts encoder", () => {
  type Opt = Record<string, unknown>;
  type St = {
    ghost_mode: string;
    ghost_mode_seconds_left: number | null;
    ghosts: { name: string; state: string; dir: string; steps: number | null }[];
    options: Record<string, Opt>;
  };
  const detail = (s: GameState) => ENCODERS["features-ghosts"].encode(s, findDecisionPoint(s));
  const plain = (s: GameState) => (ENCODERS.features.encode(s, findDecisionPoint(s)).state as St).options;

  it("keeps every features fact for each option", () => {
    const s = playing();
    for (let i = 0; i < 400; i++) step(s);
    const d = (detail(s).state as St).options;
    for (const [k, o] of Object.entries(plain(s))) expect(d[k]).toMatchObject(o);
  });

  it("adds the second nearest normal ghost on a path", () => {
    const s = playing(); // junction (12,23), exits left and up
    Object.assign(s.ghosts[0], { x: 12, y: 21, state: "active", frightened: false, dir: "down" });
    Object.assign(s.ghosts[1], { x: 12, y: 22, state: "active", frightened: false, dir: "up" });
    const up = (detail(s).state as St).options.up;
    expect(up.ghost_coming).toBe(false); // nearest, at (12,22), heads away
    expect(up.ghost2 as number).toBeGreaterThan(up.ghost as number);
    expect(up.ghost2_coming).toBe(true);
    expect(up).not.toHaveProperty("ghost3");
  });

  it("lists all four ghosts with name, state, heading, and shortest steps", () => {
    const s = playing();
    Object.assign(s.ghosts[1], { x: 12, y: 21, state: "active", frightened: true, dir: "down" });
    const st = detail(s).state as St;
    expect(st.ghosts.map((g) => g.name)).toEqual(s.ghosts.map((g) => g.name));
    expect(st.ghosts.map((g) => g.dir)).toEqual(s.ghosts.map((g) => g.dir));
    expect(st.ghosts[1].state).toBe("frightened");
    expect(st.ghosts[1].steps).not.toBeNull();
    s.ghosts.forEach((g, i) => {
      if (g.state === "house") expect(st.ghosts[i].state).toBe("in_house");
    });
    const normal = st.ghosts.filter((g) => g.state === "normal" && g.steps !== null && g.steps <= 15);
    const nearestOnOptions = Object.values(st.options).map((o) => o.ghost as number | null).filter((v) => v !== null);
    if (normal.length) expect(Math.min(...normal.map((g) => g.steps!))).toBe(Math.min(...nearestOnOptions));
  });

  it("reports the ghost mode and when ghosts next turn around, with short instructions", () => {
    const s = playing();
    const enc = detail(s);
    const st = enc.state as St;
    expect(st.ghost_mode).toBe("scatter");
    expect(st.ghost_mode_seconds_left).toBeGreaterThan(0);
    expect(st.ghost_mode_seconds_left).toBeLessThanOrEqual(7);
    for (const field of ["ghost2", "ghosts", "dir", "ghost_mode"]) expect(enc.instructions).toContain(field);
    const base = ENCODERS.features.encode(s, findDecisionPoint(s)).instructions;
    expect(enc.instructions.length - base.length).toBeLessThan(500);
  });

  it("reports facts and never verdicts, within the token budget", () => {
    const s = playing();
    for (let i = 0; i < 600; i++) step(s);
    const text = JSON.stringify(detail(s).state);
    expect(text).not.toMatch(/safe|danger|recommend|best|score/i);
    expect(text.length).toBeLessThan(1100);
  });
});
