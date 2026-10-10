import { describe, expect, it } from "vitest";
import { makePolicy } from "../src/agent/factory.ts";
import { randomWalkPolicy } from "../src/agent/policies.ts";
import type { EncodedDecision } from "../src/encoders/types.ts";
import type { GameState } from "../src/engine/types.ts";

const enc = (junction: string, keys: string[]) => ({ keys, decision: { key: junction } }) as unknown as EncodedDecision;
const at = (tick: number) => ({ tick }) as GameState;

describe("randomWalkPolicy", () => {
  it("keeps one exit per junction while it is asked again", async () => {
    const p = randomWalkPolicy(7, 30);
    await p.decide(enc("5,5", ["up", "left", "back"]), undefined, at(0)); // may be the once-a-second draw
    const first = (await p.decide(enc("5,5", ["up", "left", "back"]), undefined, at(1))).choice;
    expect(first).not.toBe("back");
    for (let tick = 2; tick < 30; tick++) {
      expect((await p.decide(enc("5,5", ["up", "left", "back"]), undefined, at(tick))).choice).toBe(first);
    }
  });

  it("turns back with probability 1/4, at most once per interval", async () => {
    const p = randomWalkPolicy(3, 30);
    let backs = 0;
    for (let tick = 0; tick < 3000; tick++) {
      const d = await p.decide(enc(`j${Math.floor(tick / 10)}`, ["up", "left", "back"]), undefined, at(tick));
      if (d.choice === "back") backs++;
    }
    // 100 draws, one per 30 ticks, each turning back with probability 1/4.
    expect(backs).toBeGreaterThan(12);
    expect(backs).toBeLessThan(38);
  });

  it("picks every exit sometimes", async () => {
    const p = randomWalkPolicy(11, 1e9);
    const seen = new Set<string>();
    for (let j = 0; j < 200; j++) seen.add((await p.decide(enc(`j${j}`, ["up", "left", "right", "back"]), undefined, at(j))).choice);
    expect(seen).toEqual(new Set(["up", "left", "right"]));
  });

  it("flips again after a game restart resets the tick", async () => {
    const p = randomWalkPolicy(5, 30);
    for (let tick = 0; tick < 3000; tick++) await p.decide(enc("a", ["up", "back"]), undefined, at(tick));
    let backs = 0;
    // Shorter than the first game, so without the reset no flip would come due.
    for (let tick = 0; tick < 2900; tick++) {
      if ((await p.decide(enc(`b${tick}`, ["up", "back"]), undefined, at(tick))).choice === "back") backs++;
    }
    expect(backs).toBeGreaterThan(0);
  });

  it("is built by name with a custom interval, without calling a model", async () => {
    for (const name of ["random-1s", "random-2.5s"]) {
      const p = makePolicy(name, 1, { decisionBaseUrl: "http://127.0.0.1:9" });
      expect(p.name).toBe(name);
      expect(["up", "back"]).toContain((await p.decide(enc("a", ["up", "back"]), undefined, at(0))).choice);
    }
  });
});
