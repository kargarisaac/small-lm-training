from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import chatgpt_subscription_shim
from common import config as cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GPT through the local ChatGPT subscription shim.")
    parser.add_argument("--model", default=cfg.GPT_TEACHER_MODEL)
    parser.add_argument("--reasoning-effort", default=cfg.GPT_TEACHER_REASONING_EFFORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    chatgpt_subscription_shim.serve_chatgpt_subscription(args.model, args.host, args.port, args.reasoning_effort)


if __name__ == "__main__":
    main()
