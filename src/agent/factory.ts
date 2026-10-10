import { greedyPolicy, randomByName, systemOnePolicy, type Policy } from "./policies.ts";
import { oraclePolicy } from "./oracle.ts";
import { llmPolicy } from "./llm.ts";
import { decisionsPolicy, DECISIONS_DEFAULT_BASE_URL } from "./decisions.ts";
import { teacherPolicy, type TeacherDecision } from "./teacher.ts";

export interface PolicyEnv {
  decisionBaseUrl: string;
  teacherBaseUrl?: string;
  teacherModel?: string;
  /** TEACHER_API_KEY, for hosted endpoints. */
  teacherApiKey?: string;
  /** TEACHER_EXTRA_BODY parsed as JSON: fields merged into every teacher request. */
  teacherExtraBody?: Record<string, unknown>;
  /** TEACHER_MAX_RETRIES: retries after HTTP 429 or 5xx (default 0). */
  teacherMaxRetries?: number;
  /** Shared by every teacher policy made with this env: identical prompts are asked once. */
  teacherCache?: Map<string, TeacherDecision>;
}

/**
 * Policies by name: "random", "random-1s" (one random exit per junction; once per second, go on or turn back
 * at 3:1; "random-2.5s" sets the interval), "greedy", "jev" (TypeSafe's hosted Jev), "openai:<model>" (OpenAI's Decisions API,
 * e.g. "openai:gpt-6-luna"), "oracle" (engine rollouts; "oracle-8s-x2"
 * sets an 8 s horizon and 2 samples), "teacher", "teacher-think", and "-peek5s" variants that also show a 5 s simulated future per option (the
 * chat model from TEACHER_BASE_URL / TEACHER_MODEL), "llm:<model>" (a plain Ollama chat model answering in JSON
 * constrained to the legal options), or any decision model name served on the /v1/systemone endpoint.
 */
export function makePolicy(name: string, seed: number, env: PolicyEnv): Policy {
  const random = randomByName(name, seed);
  if (random) return random;
  if (name === "greedy") return greedyPolicy();
  if (name === "jev") {
    // TypeSafe's hosted API speaks the same /v1/systemone protocol as Ollama.
    if (!process.env.TYPESAFE_API_KEY) throw new Error("Set TYPESAFE_API_KEY to call Jev.");
    return systemOnePolicy({
      baseUrl: process.env.JEV_BASE_URL ?? "https://api.typesafe.ai",
      model: process.env.JEV_MODEL ?? "jev-latest",
      apiKey: process.env.TYPESAFE_API_KEY,
    });
  }
  if (name.startsWith("openai:")) {
    // OpenAI's Decisions API (/v1/decisions); gpt-6-luna is the model it serves.
    if (!process.env.OPENAI_API_KEY) throw new Error("Set OPENAI_API_KEY to call the OpenAI Decisions API.");
    return decisionsPolicy({
      baseUrl: DECISIONS_DEFAULT_BASE_URL,
      model: name.slice("openai:".length),
      apiKey: process.env.OPENAI_API_KEY,
    });
  }
  const oracle = /^oracle(?:-(\d+(?:\.\d+)?)s)?(?:-x(\d+))?$/.exec(name);
  if (oracle) return oraclePolicy({ horizonSeconds: oracle[1] ? Number(oracle[1]) : 5, samples: oracle[2] ? Number(oracle[2]) : 1 });
  if (name.startsWith("llm:")) return llmPolicy({ baseUrl: env.decisionBaseUrl, model: name.slice("llm:".length) });
  const teacher = /^teacher(-think)?(?:-peek(\d+(?:\.\d+)?)s)?$/.exec(name);
  if (teacher) {
    if (!env.teacherBaseUrl || !env.teacherModel) {
      throw new Error("Set TEACHER_BASE_URL and TEACHER_MODEL (see .env.example) to use the teacher.");
    }
    return teacherPolicy({
      baseUrl: env.teacherBaseUrl,
      model: env.teacherModel,
      apiKey: env.teacherApiKey,
      extraBody: env.teacherExtraBody,
      maxRetries: env.teacherMaxRetries,
      think: !!teacher[1],
      peekSeconds: teacher[2] ? Number(teacher[2]) : undefined,
      cache: env.teacherCache,
    });
  }
  return systemOnePolicy({ baseUrl: env.decisionBaseUrl, model: name });
}

/** Reads a local .env if present, without overriding variables already set. */
export function loadDotEnv(): void {
  try {
    (process as unknown as { loadEnvFile?: (p?: string) => void }).loadEnvFile?.(".env");
  } catch {
    // no .env
  }
}

export function policyEnvFromProcess(): PolicyEnv {
  return {
    decisionBaseUrl: process.env.VITE_DECISION_BASE_URL ?? "http://localhost:11434",
    teacherBaseUrl: process.env.TEACHER_BASE_URL,
    teacherModel: process.env.TEACHER_MODEL,
    teacherApiKey: process.env.TEACHER_API_KEY || undefined,
    teacherExtraBody: parseExtraBody(process.env.TEACHER_EXTRA_BODY),
    teacherMaxRetries: process.env.TEACHER_MAX_RETRIES ? Number(process.env.TEACHER_MAX_RETRIES) : undefined,
  };
}

function parseExtraBody(raw: string | undefined): Record<string, unknown> | undefined {
  if (!raw?.trim()) return undefined;
  const parsed: unknown = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("TEACHER_EXTRA_BODY must be a JSON object");
  return parsed as Record<string, unknown>;
}
