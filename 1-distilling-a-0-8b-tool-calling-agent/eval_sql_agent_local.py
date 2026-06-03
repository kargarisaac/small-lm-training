from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import sql_agent


def load_model(model_name: str, adapter_path: Path | None, max_seq_length: int, dtype: str, experts_implementation: str | None) -> tuple[Any, Any]:
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter path: {adapter_path}")
    tokenizer_source = adapter_path if adapter_path is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    if experts_implementation is not None:
        expert_configs = {}
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_experts_implementation"):
                expert_configs[id(config)] = config
        for config in expert_configs.values():
            config._experts_implementation = experts_implementation
    model.eval()
    if hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = max(max_seq_length, int(model.config.max_position_embeddings or 0))
    return model, tokenizer


def make_local_generator(
    *,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float,
) -> sql_agent.Generate:
    def generate(messages: list[dict[str, str]]) -> str:
        rendered_messages = sql_agent.render_baml_sql_agent_messages(messages)
        prompt = tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.QWEN_ENABLE_THINKING,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        if input_ids.shape[-1] + max_new_tokens > max_seq_length:
            raise RuntimeError(
                f"Prompt tokens ({input_ids.shape[-1]}) + max new tokens ({max_new_tokens}) exceed max seq length ({max_seq_length})."
            )
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)
        new_tokens = generated[0, input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local HF/PEFT model in the deterministic SQL-agent harness.")
    parser.add_argument("--model", default=cfg.UNSLOTH_STUDENT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR / "sql_agent_bird_critic")
    parser.add_argument("--partition", choices=["train", "eval"], default="eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=cfg.SFT_MAX_SEQ_LENGTH)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--experts-implementation", choices=["eager", "batched_mm", "grouped_mm"], default=None)
    parser.add_argument("--task-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = sql_agent.load_rows(args.data_dir, args.partition, args.limit)
    adapter_slug = cfg.filename_slug(str(args.adapter_path)) if args.adapter_path else "base"
    output_path = args.output or cfg.OUTPUT_DIR / f"{cfg.filename_slug(args.model)}_{adapter_slug}_sql_agent_{args.partition}_local_eval.json"
    results = []
    if output_path.exists():
        results = cfg.make_json_safe(json.loads(output_path.read_text(encoding="utf-8")).get("results", []))
        print(f"Loaded {len(results)} cached results from: {output_path}", flush=True)
    done = {row["id"] for row in results}

    print("Harness LLM call path: local HF/PEFT model with BAML-rendered messages", flush=True)
    print("Model:", args.model, flush=True)
    print("Adapter:", args.adapter_path or "(none)", flush=True)
    print("Experts implementation:", args.experts_implementation or "(model default)", flush=True)
    print("Rows:", len(rows), flush=True)
    print("Max turns:", args.max_turns, flush=True)
    model, tokenizer = load_model(args.model, args.adapter_path, args.max_seq_length, args.dtype, args.experts_implementation)
    generate = make_local_generator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        temperature=args.temperature,
    )

    for index, row in enumerate(rows, start=1):
        if row["id"] in done:
            print(f"{index}/{len(rows)} {row['id']} cached", flush=True)
            continue
        try:
            result = sql_agent.run_task_with_timeout(
                row,
                data_dir=args.data_dir,
                generate=generate,
                max_turns=args.max_turns,
                timeout_seconds=args.task_timeout_seconds,
            )
        except Exception as error:
            result = {
                "id": row["id"],
                "db_id": row["db_id"],
                "category": row["category"],
                "success": False,
                "stop_reason": "runtime_error",
                "turns": 0,
                "trace": [
                    {
                        "turn": 0,
                        "stop_reason": "runtime_error",
                        "error_type": type(error).__name__,
                        "error": str(error)[:4000],
                    }
                ],
            }
        results.append(result)
        summary = sql_agent.summarize_results(results)
        cfg.write_json(output_path, {"summary": summary.__dict__, "results": results, "config": vars(args)})
        print(f"{index}/{len(rows)} {row['id']} success={result['success']} stop={result['stop_reason']} turns={result['turns']} running_success={summary.success}/{summary.total}", flush=True)

    summary = sql_agent.summarize_results(results)
    cfg.write_json(output_path, {"summary": summary.__dict__, "results": results, "config": vars(args)})
    print()
    print(f"Success: {summary.success}/{summary.total} = {summary.success_rate:.3f}")
    print(f"Submitted: {summary.submitted}/{summary.total}")
    print(f"Parse failures: {summary.parse_failures}")
    print(f"Max-turn failures: {summary.max_turn_failures}")
    print(f"Repeated-action failures: {summary.repeated_action_failures}")
    print(f"Runtime errors: {summary.runtime_errors}")
    print(f"Average turns: {summary.average_turns:.2f}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
