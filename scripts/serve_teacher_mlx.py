#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Serve the Qwen3.5 35B-A3B teacher with MLX-LM."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough_args = parse_args()
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-tokens",
        str(args.max_tokens),
        "--log-level",
        args.log_level,
        *passthrough_args,
    ]

    print("Starting MLX teacher server")
    print("Model:", args.model)
    print("Health:", f"http://{args.host}:{args.port}/health")
    print("Completions:", f"http://{args.host}:{args.port}/v1/completions")
    print("Command:", " ".join(command))
    print()
    sys.stdout.flush()

    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
