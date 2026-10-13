"""Tests for systemone_vllm.py against a fake vLLM server: cd training && python3 -m unittest test_systemone_vllm"""
import json
import math
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prompt import messages
from systemone_vllm import Scorer, confidence, make_handler

LETTER_IDS = {"A": 900, "B": 901, "C": 902, "D": 903, "E": 904, "F": 905, "G": 906, "H": 907}
LOGPROB = {900: -1.0, 901: -2.0, 902: -0.5}  # what the fake model gives each letter at the answer position
SEEN = {}


class FakeVllm(BaseHTTPRequestHandler):
    served = "student-a"

    def reply(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"data": [{"id": self.served}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] != self.served:
            self.reply({"error": "model not found"}, 404)
        elif self.path == "/tokenize" and "messages" in body:
            SEEN["tokenize"] = body
            self.reply({"tokens": [1, 2, 3, 4, 5]})
        elif self.path == "/tokenize":
            self.reply({"tokens": [LETTER_IDS[body["prompt"]]]})
        elif self.path == "/v1/completions":
            SEEN["completions"] = body
            choices = []
            for i, p in enumerate(body["prompt"]):
                lid = p[-1]
                choices.append({"index": i, "text": "x", "prompt_logprobs": [None] * (len(p) - 1) + [{str(lid): {"logprob": LOGPROB[lid], "rank": 1}}]})
            self.reply({"choices": list(reversed(choices))})  # order must not matter
        else:
            self.reply({"error": "no route"}, 404)

    def log_message(self, format, *args):  # noqa: A002
        pass


def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class SystemOneVllmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vllm, vllm_url = serve(FakeVllm)
        cls.adapter, cls.url = serve(make_handler(Scorer(vllm_url)))

    @classmethod
    def tearDownClass(cls):
        for server in (cls.adapter, cls.vllm):
            server.shutdown()
            server.server_close()

    def ask(self, model="student-a"):
        body = {"model": model, "keep_alive": -1, "state": {"lives": 3},
                "questions": {"move": {"type": "choice", "instructions": "rules", "criteria": {"up": None, "left": None, "back": "turn around now"}}}}
        req = urllib.request.Request(self.url + "/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    def test_answers_like_ollama_with_a_softmax_over_the_letters(self):
        out = self.ask()
        move = out["answers"]["move"]
        z = sum(math.exp(v) for v in (-1.0, -2.0, -0.5))
        expected = {"up": math.exp(-1.0) / z, "left": math.exp(-2.0) / z, "back": math.exp(-0.5) / z}
        for k, v in expected.items():
            self.assertAlmostEqual(move["probabilities"][k], v, places=9)
        self.assertEqual(move["choice"], "back")
        self.assertAlmostEqual(move["confidence"], confidence(list(expected.values())), places=9)
        self.assertEqual(out["usage"]["input_tokens"], 5)

    def test_sends_the_training_messages_with_thinking_off_and_one_letter_per_prompt(self):
        self.ask()
        tok = SEEN["tokenize"]
        self.assertEqual(tok["messages"], messages({"lives": 3}, "rules", {"up": None, "left": None, "back": "turn around now"}, "move"))
        self.assertEqual(tok["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(tok["add_generation_prompt"])
        self.assertEqual(SEEN["completions"]["prompt"], [[1, 2, 3, 4, 5, 900], [1, 2, 3, 4, 5, 901], [1, 2, 3, 4, 5, 902]])

    def test_unknown_model_is_a_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.ask("nope")
        self.assertEqual(cm.exception.code, 404)

    def test_confidence_matches_ollama(self):
        # Two answers Ollama gave in this project: (0.87, 0.13) -> 0.441, (0.44, 0.36, 0.20) -> 0.04.
        self.assertAlmostEqual(confidence([0.8694593459291, 0.1305406540708999]), 0.441, places=3)
        self.assertAlmostEqual(confidence([0.44, 0.36, 0.20]), 0.043, places=2)

    def test_routes_by_model_across_servers(self):
        other_handler = type("OtherVllm", (FakeVllm,), {"served": "student-b"})
        other, other_url = serve(other_handler)
        try:
            scorer = Scorer([other_url, f"http://127.0.0.1:{self.vllm.server_address[1]}"])
            self.assertEqual(scorer.server_for("student-a"), f"http://127.0.0.1:{self.vllm.server_address[1]}")
            self.assertEqual(scorer.server_for("student-b"), other_url)
            self.assertEqual(sorted(m["name"] for m in scorer.tags()["models"]), ["student-a", "student-b"])
        finally:
            other.shutdown()
            other.server_close()

    def test_model_list(self):
        out = json.loads(urllib.request.urlopen(self.url + "/api/tags").read())
        self.assertEqual(out["models"][0]["name"], "student-a")


if __name__ == "__main__":
    unittest.main()
