from __future__ import annotations

from typing import Any
import random

from .qwen_format import qwen_text_from_tool_call_parts
from .tau_runtime import retail_agent_system_prompt


def _serialized_tau_message_to_qwen_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = message.get("role")
    if role == "system":
        return {"role": "system", "content": message.get("content") or ""}
    if role == "tool":
        return {"role": "tool", "content": message.get("content") or ""}
    if role == "user":
        return {"role": "user", "content": message.get("content") or ""}
    if role == "assistant":
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            content = "\n".join(
                qwen_text_from_tool_call_parts(call["name"], call.get("arguments") or {})
                for call in tool_calls
            )
            return {"role": "assistant", "content": content}
        return {"role": "assistant", "content": message.get("content") or ""}
    return None


def extract_tau_bench_retail_sft_rows_from_task_result(
    task_result: dict[str, Any],
    *,
    system_prompt: str,
    qwen_tools: list[dict[str, Any]],
    only_successful_tasks: bool = True,
    include_natural_language_targets: bool = True,
    include_tool_call_targets: bool = True,
) -> list[dict[str, Any]]:
    if only_successful_tasks and not task_result.get("is_success"):
        return []

    simulation = task_result.get("simulation") or {}
    messages = simulation.get("messages") or []
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    rows: list[dict[str, Any]] = []

    for message_index, message in enumerate(messages):
        qwen_message = _serialized_tau_message_to_qwen_message(message)
        if qwen_message is None:
            continue

        raw_qwen_output = (message.get("raw_data") or {}).get("raw_qwen_output")
        is_generated_assistant_message = qwen_message["role"] == "assistant" and raw_qwen_output is not None
        if is_generated_assistant_message:
            has_tool_call = bool(message.get("tool_calls"))
            if (has_tool_call and include_tool_call_targets) or (
                not has_tool_call and include_natural_language_targets
            ):
                rows.append(
                    {
                        "messages": [*history, qwen_message],
                        "tools": qwen_tools,
                        "task_id": str(task_result.get("task_id")),
                        "source_message_index": message_index,
                        "is_tool_call_target": has_tool_call,
                        "target_text": qwen_message["content"],
                        "source_reward": task_result.get("reward"),
                        "source_termination_reason": task_result.get("termination_reason"),
                    }
                )

        history.append(qwen_message)

    return rows


def extract_tau_bench_retail_sft_rows_from_eval_payload(
    payload: dict[str, Any],
    *,
    domain_policy: str,
    qwen_tools: list[dict[str, Any]],
    only_successful_tasks: bool = True,
    include_natural_language_targets: bool = True,
    include_tool_call_targets: bool = True,
) -> list[dict[str, Any]]:
    system_prompt = retail_agent_system_prompt(domain_policy)
    rows: list[dict[str, Any]] = []
    for task_result in payload.get("task_results", []):
        rows.extend(
            extract_tau_bench_retail_sft_rows_from_task_result(
                task_result,
                system_prompt=system_prompt,
                qwen_tools=qwen_tools,
                only_successful_tasks=only_successful_tasks,
                include_natural_language_targets=include_natural_language_targets,
                include_tool_call_targets=include_tool_call_targets,
            )
        )
    return rows


def split_sft_rows_by_task_id(
    rows: list[dict[str, Any]],
    *,
    validation_task_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    unique_task_ids = sorted({str(row["task_id"]) for row in rows})
    if not unique_task_ids:
        return [], [], set()

    shuffled_task_ids = unique_task_ids[:]
    random.Random(seed).shuffle(shuffled_task_ids)
    validation_task_count = max(1, round(len(shuffled_task_ids) * validation_task_fraction))
    validation_task_ids = set(shuffled_task_ids[:validation_task_count])

    train_rows = [row for row in rows if str(row["task_id"]) not in validation_task_ids]
    validation_rows = [row for row in rows if str(row["task_id"]) in validation_task_ids]
    return train_rows, validation_rows, validation_task_ids


def mlx_chat_row_token_length(row: dict[str, Any], tokenizer: Any) -> int:
    return len(
        tokenizer.apply_chat_template(
            row["messages"],
            tools=row.get("tools"),
            return_dict=False,
        )
    )
