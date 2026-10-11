import { describe, expect, it } from "vitest";
import { ReverseLock } from "../src/agent/loop.ts";
import type { Policy } from "../src/agent/policies.ts";
import { findDecisionPoint } from "../src/engine/decision.ts";
import { createGame, step } from "../src/engine/game.ts";
import type { GameState } from "../src/engine/types.ts";
import { ENCODERS, withoutBack } from "../src/encoders/index.ts";
import { playGame } from "../src/sim/runner.ts";

function playing(): GameState {
  const s = createGame(1);
  while (s.phase !== "playing") step(s);
  return s;
}

describe("withoutBack", () => {
  it("drops back from the keys, the criteria, and the encoded options, and leaves the input alone", () => {
    const s = playing();
    const enc = ENCODERS.features.encode(s, findDecisionPoint(s));
    const out = withoutBack(enc);
    expect(out.keys).toEqual(enc.keys.filter((k) => k !== "back"));
    expect(out.criteria).not.toHaveProperty("back");
    expect((out.state as { options: object }).options).not.toHaveProperty("back");
    expect(enc.keys).toContain("back");
    expect((enc.state as { options: object }).options).toHaveProperty("back");
  });

  it("works on encoders whose state is plain text", () => {
    const s = playing();
    const out = withoutBack(ENCODERS["ascii-window"].encode(s, findDecisionPoint(s)));
    expect(out.keys).not.toContain("back");
    expect(typeof out.state).toBe("string");
  });
});

describe("ReverseLock", () => {
  const at = (tick: number, key: string) => [{ tick } as GameState, key] as const;

  it("never blocks when off", () => {
    const lock = new ReverseLock(0);
    expect(lock.blocked(...at(0, "a"))).toBe(false);
    expect(lock.blocked(...at(1, "b"))).toBe(false);
  });

  it("blocks back for the window after the next junction changes, then lets it through", () => {
    const lock = new ReverseLock(30);
    expect(lock.blocked(...at(0, "a"))).toBe(false); // first junction seen: nothing changed yet
    expect(lock.blocked(...at(10, "a"))).toBe(false);
    expect(lock.blocked(...at(20, "b"))).toBe(true); // turned back or passed a junction
    expect(lock.blocked(...at(49, "b"))).toBe(true);
    expect(lock.blocked(...at(50, "b"))).toBe(false);
  });

  it("starts over when a new game resets the tick", () => {
    const lock = new ReverseLock(30);
    lock.blocked(...at(100, "a"));
    lock.blocked(...at(200, "b"));
    expect(lock.blocked(...at(5, "b"))).toBe(false);
  });
});

describe("playGame reverseLockSeconds", () => {
  it("does not offer back again within the window after a junction change", async () => {
    const asked: { tick: number; key: string; back: boolean }[] = [];
    const alwaysBack: Policy = {
      name: "always-back",
      async decide(enc, _signal, game) {
        asked.push({ tick: game!.tick, key: findDecisionPoint(game!).key, back: enc.keys.includes("back") });
        return { choice: enc.keys.includes("back") ? "back" : enc.keys[0], latencyMs: 0 };
      },
    };
    await playGame({ seed: 1000, policy: alwaysBack, encoder: ENCODERS.features, clock: "lockstep", maxSeconds: 20, reverseLockSeconds: 1 });
    let lastChange = -Infinity;
    for (let i = 1; i < asked.length; i++) {
      if (asked[i].key !== asked[i - 1].key) lastChange = asked[i].tick;
      if (asked[i].tick - lastChange < 30) expect(asked[i].back).toBe(false);
    }
    expect(asked.some((a) => a.back)).toBe(true);
  });
});
