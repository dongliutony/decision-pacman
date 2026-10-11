import type { DecisionPoint } from "../engine/decision.ts";
import type { Dir, GameState, StepInput } from "../engine/types.ts";

export type OptionKey = Dir | "back";

export interface EncodedDecision {
  encoder: string;
  /** Sent as the /v1/systemone `state` (string or JSON). */
  state: unknown;
  instructions: string;
  /** Option name to description (null lets the name describe itself). */
  criteria: Record<string, string | null>;
  keys: OptionKey[];
  decision: DecisionPoint;
}

export interface Encoder {
  name: string;
  encode(s: GameState, dp: DecisionPoint): EncodedDecision;
}

/** Junction exits plus "back" when Pac-Man is moving. */
export function optionKeys(dp: DecisionPoint): OptionKey[] {
  const keys: OptionKey[] = [...dp.exits];
  if (dp.back) keys.push("back");
  return keys;
}

/** The same decision without the "back" option: dropped from the keys, the criteria, and an encoded `options` object. */
export function withoutBack(enc: EncodedDecision): EncodedDecision {
  const { back: _, ...criteria } = enc.criteria;
  let state = enc.state;
  if (state && typeof state === "object" && "options" in state) {
    const { back: __, ...options } = (state as { options: Record<string, unknown> }).options;
    state = { ...state, options };
  }
  return { ...enc, state, criteria, keys: enc.keys.filter((k) => k !== "back") };
}

/** Maps a chosen option to engine input: exits become turns planned at the junction tile, "back" turns around now. */
export function optionToInput(key: string, dp: DecisionPoint): StepInput | null {
  if (key === "back") return dp.back ? { reverse: true } : null;
  if (key === "up" || key === "down" || key === "left" || key === "right") return { turn: { x: dp.x, y: dp.y, dir: key } };
  return null;
}
