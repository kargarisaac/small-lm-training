#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import make_sampler


DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
DEFAULT_STOP_STRINGS = ["<|im_end|>", "<|endoftext|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve raw Qwen text from the MLX teacher without tool-call post-processing."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


def first_stop_index(text: str, stop_strings: list[str]) -> int | None:
    indexes = [text.find(stop) for stop in stop_strings if stop and stop in text]
    if not indexes:
        return None
    return min(indexes)


class RawTeacherRuntime:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        print("Loading MLX teacher model:", model_name)
        sys.stdout.flush()
        self.model, self.tokenizer = load(model_name)
        self.lock = threading.Lock()
        print("MLX raw teacher model loaded.")
        sys.stdout.flush()

    def generate(self, payload: dict[str, Any], default_max_tokens: int) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Request JSON must include a non-empty string field: prompt")

        max_tokens = int(payload.get("max_tokens") or default_max_tokens)
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 0.0))
        top_k = int(payload.get("top_k", 0))
        stop_strings = payload.get("stop") or DEFAULT_STOP_STRINGS
        if not isinstance(stop_strings, list) or not all(isinstance(item, str) for item in stop_strings):
            raise ValueError("Request field stop must be a list of strings")

        seed = payload.get("seed")
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

        text_parts: list[str] = []
        finish_reason = "length"
        prompt_tokens = 0
        generation_tokens = 0
        started_at = time.perf_counter()

        with self.lock:
            if seed is not None:
                mx.random.seed(int(seed))

            for chunk in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if chunk.text:
                    text_parts.append(chunk.text)
                    generated_text = "".join(text_parts)
                    stop_index = first_stop_index(generated_text, stop_strings)
                    if stop_index is not None:
                        generated_text = generated_text[:stop_index]
                        finish_reason = "stop"
                        return {
                            "model": self.model_name,
                            "text": generated_text,
                            "finish_reason": finish_reason,
                            "usage": {
                                "prompt_tokens": int(chunk.prompt_tokens),
                                "completion_tokens": int(chunk.generation_tokens),
                                "total_tokens": int(chunk.prompt_tokens + chunk.generation_tokens),
                            },
                            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
                        }

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                    prompt_tokens = int(chunk.prompt_tokens)
                    generation_tokens = int(chunk.generation_tokens)

        return {
            "model": self.model_name,
            "text": "".join(text_parts),
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generation_tokens,
                "total_tokens": prompt_tokens + generation_tokens,
            },
            "elapsed_seconds": round(time.perf_counter() - started_at, 4),
        }


def make_handler(runtime: RawTeacherRuntime, default_max_tokens: int) -> type[BaseHTTPRequestHandler]:
    class RawTeacherHandler(BaseHTTPRequestHandler):
        server_version = "MLXRawTeacher/0.1"

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(200, {"status": "ok", "model": runtime.model_name})
                return
            self.send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path not in {"/generate", "/v1/raw-completions"}:
                self.send_json(404, {"error": "not_found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8")) if body else {}
                response = runtime.generate(payload, default_max_tokens=default_max_tokens)
            except Exception as error:
                self.send_json(400, {"error": str(error)})
                return

            self.send_json(200, response)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), format % args))

    return RawTeacherHandler


def main() -> None:
    args = parse_args()
    runtime = RawTeacherRuntime(args.model)
    handler = make_handler(runtime, default_max_tokens=args.max_tokens)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print("Starting MLX raw teacher server")
    print("Model:", args.model)
    print("Health:", f"http://{args.host}:{args.port}/health")
    print("Raw generation:", f"http://{args.host}:{args.port}/generate")
    print()
    sys.stdout.flush()

    server.serve_forever()


if __name__ == "__main__":
    main()
