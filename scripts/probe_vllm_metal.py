from __future__ import annotations

import argparse
import time

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError as error:
    raise SystemExit(
        "vLLM-Metal is installed in /Users/kargarisaac/.venv-vllm-metal, not the project uv env.\n"
        "Run this probe with:\n"
        "  /Users/kargarisaac/.venv-vllm-metal/bin/python scripts/probe_vllm_metal.py --model <model>\n"
        "or activate it first:\n"
        "  source /Users/kargarisaac/.venv-vllm-metal/bin/activate"
    ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="<|im_start|>user\nSay hi.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    args = parser.parse_args()

    print("loading", args.model, flush=True)
    started_at = time.time()
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=True,
    )
    print("loaded_seconds", round(time.time() - started_at, 2), flush=True)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
    )
    started_at = time.time()
    outputs = llm.generate([args.prompt], sampling_params)
    print("generate_seconds", round(time.time() - started_at, 2), flush=True)

    output = outputs[0]
    completion = output.outputs[0]
    print("text", repr(completion.text), flush=True)
    print("num_output_logprob_positions", len(completion.logprobs or []), flush=True)
    print("num_prompt_logprob_positions", len(output.prompt_logprobs or []), flush=True)
    if completion.logprobs:
        print("first_output_logprob_token_ids", list(completion.logprobs[0].keys())[:5], flush=True)


if __name__ == "__main__":
    main()
