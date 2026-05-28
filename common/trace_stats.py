from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .config import STUDENT_MODEL, filename_slug


__all__ = [
    "load_tau_bench_eval_trace_stats",
    "save_tau_bench_eval_trace_stats_csv",
    "plot_tau_bench_eval_trace_stats",
]


def _newest_tau_bench_eval_result_match(output_dir: Path, patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(output_dir.glob(pattern))
    clean_matches = [
        path
        for path in matches
        if not any(marker in path.name for marker in ["backup", "contaminated", "provider_error"])
    ]
    if not clean_matches:
        return None
    return max(clean_matches, key=lambda path: path.stat().st_mtime)


def _resolve_tau_bench_eval_result_files(
    output_dir: Path,
    user_simulator_model: str,
    student_model: str = STUDENT_MODEL,
) -> list[tuple[str, Path]]:
    user_simulator_slug = filename_slug(user_simulator_model)
    student_slug = filename_slug(student_model)
    result_specs = [
        (
            "base_student",
            output_dir
            / f"{student_slug}_tau3_bench_retail_test_official_student_eval_{user_simulator_slug}.json",
            [
                f"{student_slug}_tau3_bench_retail_test_official_student_eval_{user_simulator_slug}.json"
            ],
        ),
        (
            "teacher",
            output_dir
            / f"mlx_community_qwen3_5_35b_a3b_8bit_vllm_raw_tau3_bench_retail_test_official_teacher_eval_{user_simulator_slug}.json",
            [
                "mlx_community_qwen3_5_35b_a3b_8bit_vllm_raw_tau3_bench_retail_test_official_"
                f"teacher_eval_{user_simulator_slug}.json"
            ],
        ),
        (
            "trained_student",
            output_dir
            / f"{student_slug}_tau3_retail_sft_mlx_lm"
            / f"{student_slug}_tau3_retail_mlx_sft_eval_{user_simulator_slug}.json",
            [
                f"{student_slug}_tau3_retail_sft_mlx_lm/"
                f"{student_slug}_tau3_retail_mlx_sft_eval_{user_simulator_slug}.json"
            ],
        ),
    ]

    resolved: list[tuple[str, Path]] = []
    for label, preferred_path, fallback_patterns in result_specs:
        if preferred_path.exists():
            resolved.append((label, preferred_path))
            continue
        fallback = _newest_tau_bench_eval_result_match(output_dir, fallback_patterns)
        if fallback is not None:
            resolved.append((label, fallback))
    return resolved


def _tau_bench_eval_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = ((row.get("simulation") or {}).get("messages") or [])
    return [message for message in messages if isinstance(message, dict)]


def _tau_bench_generated_agent_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for message in messages:
        raw_data = message.get("raw_data") or {}
        if message.get("role") == "assistant" and isinstance(raw_data, dict) and "raw_qwen_output" in raw_data:
            generated.append(message)
    return generated


def _tau_bench_message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


def _tau_bench_parse_errors_for(message: dict[str, Any]) -> list[str]:
    raw_data = message.get("raw_data") or {}
    if not isinstance(raw_data, dict):
        return []
    parse_errors = raw_data.get("parse_errors") or []
    return [str(error) for error in parse_errors]


def _tau_bench_token_count_for(message: dict[str, Any], key: str) -> int:
    raw_data = message.get("raw_data") or {}
    if not isinstance(raw_data, dict):
        return 0
    value = raw_data.get(key, 0)
    return int(value or 0)


def _extract_tau_bench_eval_task_stats(
    label: str,
    result_path: Path,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []

    for row in payload.get("task_results", []):
        messages = _tau_bench_eval_messages(row)
        generated_agent = _tau_bench_generated_agent_messages(messages)
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        user_messages = [message for message in messages if message.get("role") == "user"]
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        tool_call_messages = [
            message for message in assistant_messages if _tau_bench_message_tool_calls(message)
        ]

        parse_error_turns = sum(
            1 for message in generated_agent if _tau_bench_parse_errors_for(message)
        )
        no_tool_call_turns = sum(
            1
            for message in generated_agent
            if any(
                "No <tool_call> block found" in error
                for error in _tau_bench_parse_errors_for(message)
            )
        )
        tool_call_count = sum(
            len(_tau_bench_message_tool_calls(message)) for message in assistant_messages
        )
        prompt_tokens_total = sum(
            _tau_bench_token_count_for(message, "prompt_tokens") for message in generated_agent
        )
        completion_tokens_total = sum(
            _tau_bench_token_count_for(message, "completion_tokens") for message in generated_agent
        )

        for message in assistant_messages:
            for tool_call in _tau_bench_message_tool_calls(message):
                tool_rows.append(
                    {
                        "run": label,
                        "task_id": row.get("task_id"),
                        "tool_name": tool_call.get("name"),
                        "success": bool(row.get("is_success")),
                        "termination_reason": row.get("termination_reason"),
                    }
                )

        task_rows.append(
            {
                "run": label,
                "path": str(result_path),
                "task_id": row.get("task_id"),
                "success": bool(row.get("is_success")),
                "reward": row.get("reward"),
                "termination_reason": row.get("termination_reason"),
                "duration_seconds": float(row.get("duration_seconds") or 0.0),
                "messages": int(row.get("message_count") or len(messages)),
                "dialogue_turns": len(assistant_messages) + len(user_messages),
                "assistant_turns": len(assistant_messages),
                "user_turns": len(user_messages),
                "tool_observation_turns": len(tool_messages),
                "model_decision_turns": len(generated_agent),
                "tool_call_turns": len(tool_call_messages),
                "tool_calls": tool_call_count,
                "assistant_text_turns": sum(
                    1
                    for message in generated_agent
                    if message.get("content") and not _tau_bench_message_tool_calls(message)
                ),
                "parse_error_turns": parse_error_turns,
                "no_tool_call_parse_error_turns": no_tool_call_turns,
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "tokens_total": prompt_tokens_total + completion_tokens_total,
                "duration_per_model_turn": (
                    float(row.get("duration_seconds") or 0.0) / len(generated_agent)
                    if generated_agent
                    else None
                ),
            }
        )

    return task_rows, tool_rows


def load_tau_bench_eval_trace_stats(
    output_dir: Path,
    user_simulator_model: str,
    student_model: str = STUDENT_MODEL,
) -> dict[str, Any]:
    import pandas as pd  # noqa: PLC0415

    result_files = _resolve_tau_bench_eval_result_files(
        output_dir,
        user_simulator_model,
        student_model=student_model,
    )
    task_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []

    for label, result_path in result_files:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_task_rows, result_tool_rows = _extract_tau_bench_eval_task_stats(
            label,
            result_path,
            payload,
        )
        task_rows.extend(result_task_rows)
        tool_rows.extend(result_tool_rows)

    stats_df = pd.DataFrame(task_rows)
    tool_df = pd.DataFrame(tool_rows)
    summary_df = _summarize_tau_bench_eval_trace_stats(stats_df)
    slowest_df = _slowest_tau_bench_eval_tasks(stats_df)
    top_tools_df = _top_tau_bench_eval_tools(tool_df)

    return {
        "result_files": result_files,
        "stats_df": stats_df,
        "tool_df": tool_df,
        "summary_df": summary_df,
        "slowest_df": slowest_df,
        "top_tools_df": top_tools_df,
    }


def _summarize_tau_bench_eval_trace_stats(stats_df: Any) -> Any:
    if stats_df.empty:
        return stats_df
    return (
        stats_df.groupby("run", dropna=False)
        .agg(
            tasks=("task_id", "count"),
            correct=("success", "sum"),
            accuracy=("success", "mean"),
            total_minutes=("duration_seconds", lambda values: values.sum() / 60),
            median_duration_s=("duration_seconds", "median"),
            mean_duration_s=("duration_seconds", "mean"),
            median_messages=("messages", "median"),
            median_dialogue_turns=("dialogue_turns", "median"),
            median_model_decision_turns=("model_decision_turns", "median"),
            median_tool_calls=("tool_calls", "median"),
            total_tool_calls=("tool_calls", "sum"),
            median_parse_error_turns=("parse_error_turns", "median"),
            total_prompt_tokens=("prompt_tokens_total", "sum"),
            total_completion_tokens=("completion_tokens_total", "sum"),
        )
        .reset_index()
    )


def _slowest_tau_bench_eval_tasks(stats_df: Any, limit: int = 15) -> Any:
    if stats_df.empty:
        return stats_df
    columns = [
        "run",
        "task_id",
        "success",
        "termination_reason",
        "duration_seconds",
        "messages",
        "model_decision_turns",
        "tool_calls",
        "parse_error_turns",
    ]
    return stats_df.sort_values("duration_seconds", ascending=False)[columns].head(limit)


def _top_tau_bench_eval_tools(tool_df: Any, limit_per_run: int = 12) -> Any:
    if tool_df.empty:
        return tool_df
    return (
        tool_df.groupby(["run", "tool_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["run", "count"], ascending=[True, False])
        .groupby("run")
        .head(limit_per_run)
    )


def save_tau_bench_eval_trace_stats_csv(stats_df: Any, output_dir: Path) -> Path | None:
    if stats_df.empty:
        return None
    stats_csv_path = output_dir / "tau3_retail_eval_trace_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    return stats_csv_path


def plot_tau_bench_eval_trace_stats(stats_df: Any) -> None:
    if stats_df.empty:
        return

    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_metrics = [
        ("duration_seconds", "Duration per task (seconds)"),
        ("messages", "Messages per task"),
        ("model_decision_turns", "Model decision turns per task"),
        ("tool_calls", "Tool calls per task"),
        ("parse_error_turns", "Parse-error turns per task"),
        ("completion_tokens_total", "Completion tokens per task"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, title) in zip(axes.flatten(), plot_metrics):
        stats_df.boxplot(column=metric, by="run", ax=ax, grid=False, rot=20)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("τ³-Bench Retail Eval Trace Statistics")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, group in stats_df.groupby("run"):
        ax.scatter(
            group["model_decision_turns"],
            group["duration_seconds"],
            label=run_name,
            alpha=0.75,
            s=55,
        )
    ax.set_xlabel("Model decision turns")
    ax.set_ylabel("Duration seconds")
    ax.set_title("Longer tasks usually mean more agent/user/environment turns")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.show()
