"""Serve Ollama's /v1/systemone decision API on top of a vLLM server.

  python3 training/systemone_vllm.py --vllm http://localhost:8000 --port 11500
  python3 training/systemone_vllm.py --vllm http://localhost:8001 --vllm http://localhost:8002   # several servers

The game, training/score_ollama.py, and DAgger play speak /v1/systemone; vLLM does
not. For each choice question this adapter:

1. renders the request with training/prompt.py, the messages the student was trained on;
2. asks vLLM to tokenize them with the model's own chat template, thinking off
   (train.py applies the same template through the Hugging Face tokenizer);
3. appends each option letter (A, B, ...) as one token and asks vLLM for that token's
   log-probability at the answer position (prompt_logprobs), all letters in one call,
   so the prompt prefix is computed once;
4. returns a softmax over those letters only, the top choice, and Ollama's confidence
   (1 minus the entropy over the maximum entropy).

That is how train.py scores and how Ollama answers. The `model` field names a vLLM
model: a base, or a LoRA adapter registered with --lora-modules. With several --vllm
servers (for example one merged model per GPU), each request goes to the server that
lists that model in /v1/models. GET /api/tags and
/api/version answer too, so the browser's model list works. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prompt import letters, messages


class VllmError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def post(url: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise VllmError(e.code, e.read().decode(errors="replace")) from None


def get(url: str, timeout: float = 30) -> dict:
    try:
        return json.loads(urllib.request.urlopen(url, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise VllmError(e.code, e.read().decode(errors="replace")) from None


def confidence(probs: list[float]) -> float:
    """Ollama's confidence: 1 - H(p) / log(n); 1 for a single option."""
    if len(probs) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1.0 - h / math.log(len(probs)))


class Scorer:
    def __init__(self, vllm: str | list[str]):
        self.servers = [u.rstrip("/") for u in ([vllm] if isinstance(vllm, str) else vllm)]
        self._letters: dict[str, list[int]] = {}
        self._routes: dict[str, str] = {}
        self._lock = threading.Lock()

    def server_for(self, model: str) -> str:
        """The vLLM server that serves `model`; with one server, that one."""
        if len(self.servers) == 1:
            return self.servers[0]
        with self._lock:
            url = self._routes.get(model)
        if url is None:
            for u in self.servers:
                if any(m["id"] == model for m in get(f"{u}/v1/models")["data"]):
                    url = u
                    break
            if url is None:
                raise VllmError(404, f"model {model} not found on {', '.join(self.servers)}")
            with self._lock:
                self._routes[model] = url
        return url

    def letter_ids(self, model: str, n: int) -> list[int]:
        """Token ids of A, B, ... for this model's tokenizer; each must be one token, as train.py asserts."""
        with self._lock:
            ids = self._letters.get(model)
        if ids is None or len(ids) < n:
            ids = []
            for ch in letters(max(n, 8)):
                toks = post(f"{self.server_for(model)}/tokenize", {"model": model, "prompt": ch, "add_special_tokens": False})["tokens"]
                if len(toks) != 1:
                    raise VllmError(500, f"letter {ch} is not a single token for {model}: {toks}")
                ids.append(toks[0])
            with self._lock:
                self._letters[model] = ids
        return ids[:n]

    def prompt_tokens(self, model: str, msgs: list[dict]) -> list[int]:
        body = {
            "model": model,
            "messages": msgs,
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return post(f"{self.server_for(model)}/tokenize", body)["tokens"]

    def choose(self, model: str, state, instructions: str, criteria: dict, field: str) -> tuple[dict, int]:
        keys = list(criteria.keys())
        if not keys:
            raise VllmError(400, f"question {field} has no options")
        ids = self.letter_ids(model, len(keys))
        toks = self.prompt_tokens(model, messages(state, instructions, criteria, field))
        out = post(f"{self.server_for(model)}/v1/completions", {
            "model": model,
            "prompt": [toks + [lid] for lid in ids],
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 1,
        })
        logps = [0.0] * len(keys)
        for c in out["choices"]:
            i = c["index"]
            last = c["prompt_logprobs"][-1]  # the appended letter's position
            entry = last.get(str(ids[i])) if isinstance(last, dict) else None
            if entry is None:
                raise VllmError(500, f"vLLM returned no log-probability for letter {letters(len(keys))[i]}")
            logps[i] = entry["logprob"] if isinstance(entry, dict) else float(entry)
        top = max(logps)
        ws = [math.exp(v - top) for v in logps]
        total = sum(ws)
        probs = [w / total for w in ws]
        best = max(range(len(keys)), key=lambda i: probs[i])
        answer = {
            "type": "choice",
            "choice": keys[best],
            "probabilities": dict(zip(keys, probs)),
            "confidence": confidence(probs),
        }
        return answer, len(toks)

    def systemone(self, body: dict) -> dict:
        model = body.get("model")
        if not model:
            raise VllmError(400, "model is required")
        answers, input_tokens = {}, 0
        for field, q in (body.get("questions") or {}).items():
            if q.get("type") != "choice":
                raise VllmError(400, f"question {field}: only choice questions are supported")
            answers[field], n = self.choose(model, body.get("state"), q.get("instructions", ""), q.get("criteria") or {}, field)
            input_tokens += n
        return {"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": len(answers)}}

    def tags(self) -> dict:
        ids = [m["id"] for u in self.servers for m in get(f"{u}/v1/models")["data"]]
        return {"models": [{"name": i, "model": i} for i in ids]}


def make_handler(scorer: Scorer):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:  # browser preflight
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:
            try:
                if self.path == "/api/version":
                    self.reply(200, {"version": "systemone-vllm"})
                elif self.path == "/api/tags":
                    self.reply(200, scorer.tags())
                else:
                    self.reply(404, {"error": f"no route {self.path}"})
            except VllmError as e:
                self.reply(502, {"error": str(e)})

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self.reply(404, {"error": f"no route {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                self.reply(200, scorer.systemone(body))
            except VllmError as e:
                # vLLM answers 404 for an unknown model; keep that, so clients report "model not found".
                status = e.status if e.status in (400, 404) else 500
                self.reply(status, {"error": str(e)})
            except (ValueError, KeyError) as e:
                self.reply(400, {"error": f"bad request: {e}"})

        def log_message(self, format, *args) -> None:  # noqa: A002  quiet: one line per request is too much at 30 Hz
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm", action="append", help="base URL of a vLLM OpenAI-compatible server; repeat for several (default http://localhost:8000)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11500)
    args = ap.parse_args()
    vllm = args.vllm or ["http://localhost:8000"]
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Scorer(vllm)))
    print(f"/v1/systemone on http://{args.host}:{args.port} -> vLLM {', '.join(vllm)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
