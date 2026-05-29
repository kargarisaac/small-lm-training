from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import retail_eval, sft_rows, tau_runtime, teacher_backends, user_simulator


DEFAULT_NVIDIA_TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TOKENIZER_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher on tau3-bench retail train tasks and extract SFT rows.")
    parser.add_argument("--provider", default="vllm_raw", choices=["vllm_raw", "chatgpt_raw", "ollama_raw", "mlx_raw_server"])
    parser.add_argument("--model", default=DEFAULT_NVIDIA_TEACHER_MODEL)
    parser.add_argument("--request-model", default=None)
    parser.add_argument("--server-base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all train tasks.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=cfg.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = cfg.setup_notebook_paths(blog_dir_name="1-distilling-a-0-8b-tool-calling-agent")
    user_simulator_runtime = user_simulator.start_tau_bench_user_simulator_from_env()
    user_model = user_simulator_runtime.model
    user_args = user_simulator_runtime.args
    server_base_url = args.server_base_url
    if args.provider == "chatgpt_raw" and server_base_url is None:
        if user_simulator_runtime.shim is None:
            raise RuntimeError("chatgpt_raw needs --server-base-url when the user simulator is not using the local shim.")
        server_base_url = user_simulator_runtime.shim.base_url

    teacher_config = cfg.TeacherConfig(
        provider=args.provider,
        server_base_url=server_base_url,
        model_name=args.model,
        request_model=args.request_model or args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        reasoning_effort=args.reasoning_effort,
        request_timeout_seconds=600,
        max_new_tokens=args.max_new_tokens,
    )
    if not teacher_backends.teacher_runtime_is_configured(teacher_config):
        raise RuntimeError("Teacher backend is not ready.")

    tokenizer = cfg.load_tokenizer(TOKENIZER_MODEL)
    retail_domain = tau_runtime.load_tau_bench_retail_domain(paths.data_dir, cfg.TAU_BENCH_REPO_REVISION)
    train_tasks = retail_domain.runtime.get_tasks("train")
    task_objects = train_tasks[: args.limit] if args.limit > 0 else train_tasks

    user_slug = cfg.filename_slug(user_model)
    teacher_slug = cfg.filename_slug(args.model)
    train_output_path = paths.output_dir / f"{teacher_slug}_{args.provider}_tau3_bench_retail_train_teacher_trajectories_{user_slug}.json"
    trace_dir = paths.output_dir / "local_traces" / train_output_path.stem
    sft_output_path = paths.output_dir / f"{teacher_slug}_{args.provider}_tau3_bench_retail_train_successful_sft_chat_rows_{user_slug}.jsonl"
    train_config = cfg.TauBenchRetailEvalConfig(
        dataset_revision=cfg.TAU_BENCH_REPO_REVISION,
        student_model_name=args.model,
        user_simulator_model=user_model,
        user_simulator_args=user_args,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        model_role="teacher_train",
    )
    runner = retail_eval.TauBenchRetailTeacherEvalRunner(
        runtime=retail_domain.runtime,
        tokenizer=tokenizer,
        qwen_tools=retail_domain.tools,
        tool_schema_by_name=retail_domain.tool_schema_by_name,
        teacher_config=teacher_config,
        config=train_config,
        trace_dir=trace_dir,
    )

    print("Teacher:", args.model)
    print("Provider:", args.provider, teacher_config.server_base_url)
    print("User simulator:", user_model)
    print("Train tasks:", len(task_objects))
    print("Trajectory output:", train_output_path)
    print("SFT rows output:", sft_output_path)

    train_payload = retail_eval.run_tau_bench_retail_eval_tasks(
        task_objects=task_objects,
        runner=runner,
        output_path=train_output_path,
        print_progress=True,
        show_progress_bar=True,
        quiet_tau2_console=True,
    )
    rows = sft_rows.extract_tau_bench_retail_sft_rows_from_eval_payload(
        train_payload,
        domain_policy=retail_domain.policy,
        qwen_tools=retail_domain.tools,
        only_successful_tasks=True,
        include_natural_language_targets=True,
        include_tool_call_targets=True,
    )
    cfg.write_jsonl(sft_output_path, rows)
    successful_tasks = {row["task_id"] for row in train_payload.get("task_results", []) if row.get("is_success")}
    tool_rows = sum(row["is_tool_call_target"] for row in rows)
    lengths = [sft_rows.mlx_chat_row_token_length(row, tokenizer) for row in rows]
    print()
    print("Teacher train successes:", len(successful_tasks), "/", len(train_payload.get("task_results", [])))
    print("SFT rows:", len(rows))
    print("Tool-call rows:", tool_rows)
    print("Natural-language rows:", len(rows) - tool_rows)
    if lengths:
        print("Token lengths:", json.dumps({
            "min": min(lengths),
            "p50": cfg.percentile_int(lengths, 0.50),
            "p90": cfg.percentile_int(lengths, 0.90),
            "p95": cfg.percentile_int(lengths, 0.95),
            "max": max(lengths),
        }, indent=2))
    print("Saved:", sft_output_path)


if __name__ == "__main__":
    main()
