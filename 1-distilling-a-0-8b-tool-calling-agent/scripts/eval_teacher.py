from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as cfg
from common import retail_eval, tau_runtime, teacher_backends, user_simulator


DEFAULT_NVIDIA_TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TOKENIZER_MODEL = "Qwen/Qwen3.5-0.8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a teacher backend on tau3-bench retail.")
    parser.add_argument("--provider", default="vllm_raw", choices=["vllm_raw", "chatgpt_raw", "ollama_raw", "mlx_raw_server"])
    parser.add_argument("--model", default=DEFAULT_NVIDIA_TEACHER_MODEL)
    parser.add_argument("--request-model", default=None)
    parser.add_argument("--server-base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all test tasks.")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=cfg.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow", action="store_true")
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
    tasks = retail_domain.runtime.get_tasks("test")
    task_objects = tasks[: args.limit] if args.limit > 0 else tasks

    user_slug = cfg.filename_slug(user_model)
    teacher_slug = cfg.filename_slug(args.model)
    output_path = paths.output_dir / f"{teacher_slug}_{args.provider}_tau3_bench_retail_test_official_teacher_eval_{user_slug}.json"
    trace_dir = paths.output_dir / "local_traces" / output_path.stem
    mlflow_config = cfg.MlflowConfig(
        enabled=args.mlflow,
        experiment_name="distillation-blogs-tau3",
        log_full_artifacts=True,
        log_spans=False,
    )
    eval_config = cfg.TauBenchRetailEvalConfig(
        dataset_revision=cfg.TAU_BENCH_REPO_REVISION,
        student_model_name=args.model,
        user_simulator_model=user_model,
        user_simulator_args=user_args,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        model_role="teacher",
    )
    runner = retail_eval.TauBenchRetailTeacherEvalRunner(
        runtime=retail_domain.runtime,
        tokenizer=tokenizer,
        qwen_tools=retail_domain.tools,
        tool_schema_by_name=retail_domain.tool_schema_by_name,
        teacher_config=teacher_config,
        config=eval_config,
        trace_dir=trace_dir,
    )

    print("Teacher:", args.model)
    print("Provider:", args.provider, teacher_config.server_base_url)
    print("User simulator:", user_model)
    print("Tasks:", len(task_objects))
    print("Output:", output_path)
    payload = retail_eval.run_tau_bench_retail_eval_tasks(
        task_objects=task_objects,
        runner=runner,
        output_path=output_path,
        print_progress=True,
        show_progress_bar=True,
        quiet_tau2_console=True,
        mlflow_config=mlflow_config,
        mlflow_run_name=output_path.stem,
        mlflow_tags={"tau3.script": Path(__file__).name, "tau3.runtime": args.provider},
    )
    rows = payload["task_results"]
    correct = sum(1 for row in rows if row["is_success"])
    failures = Counter(row["termination_reason"] for row in rows if not row["is_success"])
    print()
    print(f"Accuracy: {correct}/{len(rows)} = {correct / len(rows):.3f}" if rows else "No rows.")
    print("Failures:", json.dumps(dict(failures), indent=2))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
