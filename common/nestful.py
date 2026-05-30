from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ast
import json
import math
import random

from datasets import load_dataset

from . import config as cfg


SYSTEM_PROMPT = """You are a function-calling model solving nested API-call tasks.

You will receive a user query and a JSON catalog of available functions.
Return only a JSON array of function calls. Do not write prose.

Each function call must have this shape:
{"name": "function_name", "label": "$var_1", "arguments": {"arg_name": "value"}}

Use a new label for each call, in order: "$var_1", "$var_2", ...
When a later call needs the output of an earlier call, reference it with "$var_1.output_name$".
The output names are listed in each function's output_parameters field."""


@dataclass(frozen=True)
class EvalSummary:
    total: int
    parsed: int
    exact_correct: int
    name_sequence_correct: int
    exact_accuracy: float
    name_sequence_accuracy: float
    parse_rate: float


def prepare_data(
    output_dir: Path,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    eval_fraction: float = cfg.NESTFUL_EVAL_FRACTION,
    seed: int = cfg.NESTFUL_SPLIT_SEED,
    max_calls: int | None = None,
) -> dict[str, Any]:
    rows = [normalize_source_row(index, row) for index, row in enumerate(load_dataset(cfg.NESTFUL_DATASET, split="train"))]
    random.Random(seed).shuffle(rows)
    eval_count = max(1, math.ceil(len(rows) * eval_fraction))
    eval_rows = with_partition(filter_by_max_calls(rows[:eval_count], max_calls), "eval", eval_limit)
    train_rows = with_partition(filter_by_max_calls(rows[eval_count:], max_calls), "train", train_limit)
    cfg.write_jsonl(output_dir / "train.jsonl", train_rows)
    cfg.write_jsonl(output_dir / "eval.jsonl", eval_rows)
    stats = dataset_stats(train_rows, eval_rows, eval_fraction, seed, max_calls)
    cfg.write_json(output_dir / "stats.json", stats)
    return stats


def filter_by_max_calls(rows: list[dict[str, Any]], max_calls: int | None) -> list[dict[str, Any]]:
    if max_calls is None:
        return rows
    return [row for row in rows if len(row["expected_calls"]) <= max_calls]


def normalize_source_row(source_index: int, row: dict[str, Any]) -> dict[str, Any]:
    tools = json.loads(row["tools"])
    expected_calls = normalize_call_sequence(json.loads(row["output"]))
    return {
        "source_index": source_index,
        "source_id": row["sample_id"],
        "input": row["input"],
        "tools": tools,
        "expected_calls": expected_calls,
        "gold_answer": parse_gold_answer(row["gold_answer"]),
    }


def with_partition(rows: list[dict[str, Any]], partition: str, limit: int | None) -> list[dict[str, Any]]:
    selected = rows if limit is None else rows[:limit]
    prepared = []
    for index, row in enumerate(selected):
        row = dict(row)
        row["id"] = f"{partition}_{index:05d}"
        row["partition"] = partition
        row["messages"] = messages_for_row(row)
        prepared.append(row)
    return prepared


def messages_for_row(row: dict[str, Any], assistant_calls: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nAvailable functions JSON:\n" + json.dumps(row["tools"], ensure_ascii=False),
        },
        {"role": "user", "content": row["input"]},
    ]
    if assistant_calls is not None:
        messages.append({"role": "assistant", "content": json.dumps(assistant_calls, ensure_ascii=False)})
    elif "expected_calls" in row:
        messages.append({"role": "assistant", "content": json.dumps(row["expected_calls"], ensure_ascii=False)})
    return messages


def normalize_call_sequence(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_map: dict[str, str] = {}
    normalized = []
    for index, call in enumerate(calls, start=1):
        old_label = str(call.get("label") or f"$var_{index}")
        old_key = old_label.strip("$")
        new_label = f"$var_{index}"
        label_map[old_key] = new_label.strip("$")
        normalized.append(
            {
                "name": str(call["name"]),
                "label": new_label,
                "arguments": cfg.make_json_safe(rewrite_label_references(call.get("arguments") or {}, label_map)),
            }
        )
    return normalized


def rewrite_label_references(value: Any, label_map: dict[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        inner = value[1:-1] if value.endswith("$") else value[1:]
        if "." in inner:
            label, output_name = inner.split(".", 1)
            if label in label_map:
                return f"${label_map[label]}.{output_name}$"
    if isinstance(value, list):
        return [rewrite_label_references(item, label_map) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_label_references(item, label_map) for key, item in value.items()}
    return value


def parse_gold_answer(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def dataset_stats(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    eval_fraction: float,
    seed: int,
    max_calls: int | None,
) -> dict[str, Any]:
    sequence_lengths = [len(row["expected_calls"]) for row in train_rows + eval_rows]
    tool_counts = [len(row["tools"]) for row in train_rows + eval_rows]
    call_counts: Counter[str] = Counter()
    for row in train_rows + eval_rows:
        call_counts.update(call["name"] for call in row["expected_calls"])
    return {
        "dataset": cfg.NESTFUL_DATASET,
        "split_seed": seed,
        "eval_fraction": eval_fraction,
        "max_calls": max_calls,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "sequence_length": numeric_stats(sequence_lengths),
        "tool_count": numeric_stats(tool_counts),
        "top_called_functions": dict(call_counts.most_common(20)),
    }


def numeric_stats(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(values) / len(values),
        "p90": ordered[math.floor((len(ordered) - 1) * 0.9)],
        "max": ordered[-1],
    }


def load_prepared_rows(data_dir: Path, partition: str, limit: int | None = None) -> list[dict[str, Any]]:
    path = data_dir / f"{partition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Prepared NESTFUL file not found: {path}. Run prepare_nestful.py first.")
    return cfg.read_jsonl(path, limit)


def split_train_validation(
    rows: list[dict[str, Any]],
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No rows to split.")
    validation_count = max(1, math.ceil(len(rows) * validation_fraction))
    return rows[validation_count:], rows[:validation_count]


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return apply_chat_template(tokenizer, row["messages"][:-1], add_generation_prompt=True)


def render_training_text(tokenizer: Any, row: dict[str, Any]) -> str:
    return apply_chat_template(tokenizer, row["messages"], add_generation_prompt=False)


def tokenize_sft_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, list[int]] | None:
    prompt_text = render_prompt(tokenizer, row)
    full_text = render_training_text(tokenizer, row)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_ids) > max_length:
        return None
    if full_ids[: len(prompt_ids)] == prompt_ids:
        target_ids = full_ids[len(prompt_ids) :]
    else:
        target_ids = tokenizer.encode(full_text[len(prompt_text) :], add_special_tokens=False)
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def write_mlx_lm_data(data_dir: Path, train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_jsonl(data_dir / "train.jsonl", [{"messages": row["messages"]} for row in train_rows])
    cfg.write_jsonl(data_dir / "valid.jsonl", [{"messages": row["messages"]} for row in valid_rows])
    cfg.write_jsonl(data_dir / "test.jsonl", [{"messages": row["messages"]} for row in valid_rows])


def parse_call_sequence(text: str) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        stripped = stripped.replace(token, "").strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    value = parse_json_value_or_sequence(stripped)
    if value is None:
        return None
    if isinstance(value, dict):
        value = value["calls"] if "calls" in value else [value]
    if not isinstance(value, list):
        return None
    try:
        return normalize_call_sequence(value)
    except (KeyError, TypeError):
        return None


def parse_json_value_or_sequence(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    index = 0
    values = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return None
        values.append(value)
    return values or None


def score_output(row: dict[str, Any], raw_output: str) -> dict[str, Any]:
    predicted = parse_call_sequence(raw_output)
    expected = row["expected_calls"]
    expected_names = [call["name"] for call in expected]
    predicted_names = [call["name"] for call in predicted] if predicted is not None else None
    exact = predicted == expected
    name_sequence = predicted_names == expected_names
    return {
        "id": row["id"],
        "correct": exact,
        "exact_correct": exact,
        "name_sequence_correct": name_sequence,
        "error": None if exact else "parse_error" if predicted is None else "sequence_mismatch",
        "expected": expected,
        "predicted": predicted,
        "expected_names": expected_names,
        "predicted_names": predicted_names,
        "gold_answer": row["gold_answer"],
        "raw_output": raw_output,
    }


def execute_call_sequence(row: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    output_names_by_tool = {
        tool["name"]: list((tool.get("output_parameters") or {}).keys())
        for tool in row["tools"]
    }
    variables: dict[str, dict[str, Any]] = {}
    trace = []
    for step_index, call in enumerate(calls, start=1):
        resolved_arguments = resolve_execution_references(call.get("arguments") or {}, variables)
        output_names = output_names_by_tool.get(call["name"]) or ["result"]
        step = {
            "step": step_index,
            "model_call": call,
            "arguments_sent_to_env": resolved_arguments,
            "tool_output_names": output_names,
        }
        try:
            raw_result = run_deterministic_function(call["name"], resolved_arguments)
            env_result = wrap_execution_result(raw_result, output_names)
            variables[call["label"].strip("$")] = env_result
            step["env_result"] = env_result
        except Exception as error:
            step["env_error"] = f"{type(error).__name__}: {error}"
            trace.append(step)
            break
        trace.append(step)
    return {"trace": trace, "variables": variables}


def resolve_execution_references(value: Any, variables: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith("$") and value.endswith("$"):
        label, _, output_name = value[1:-1].partition(".")
        if label in variables and output_name in variables[label]:
            return variables[label][output_name]
    if isinstance(value, list):
        return [resolve_execution_references(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: resolve_execution_references(item, variables) for key, item in value.items()}
    return value


def run_deterministic_function(name: str, arguments: dict[str, Any]) -> Any:
    if name == "add":
        return arguments["arg_0"] + arguments["arg_1"]
    if name == "subtract":
        return arguments["arg_0"] - arguments["arg_1"]
    if name == "multiply":
        return arguments["arg_0"] * arguments["arg_1"]
    if name == "divide":
        return arguments["arg_0"] / arguments["arg_1"]
    if name == "compute_mean":
        values = [item for row in arguments["two_d_list"] for item in row]
        if not values:
            raise ValueError("compute_mean received an empty list.")
        return sum(values) / len(values)
    if name == "modular_exponentiation":
        return (arguments["a"] ** arguments["b"]) % arguments["c"]
    if name == "convert_to_ascii_codes":
        return [ord(character) for character in arguments["text"]]
    if name == "find_extremes":
        numbers = arguments["numbers"]
        return [min(numbers), max(numbers)] if numbers else []
    raise NotImplementedError(f"No deterministic notebook executor for {name!r}.")


def wrap_execution_result(value: Any, output_names: list[str]) -> dict[str, Any]:
    if len(output_names) == 1:
        return {output_names[0]: cfg.make_json_safe(value)}
    if isinstance(value, (list, tuple)) and len(value) == len(output_names):
        return {name: cfg.make_json_safe(item) for name, item in zip(output_names, value, strict=True)}
    return {"output_0": cfg.make_json_safe(value)}


def teacher_sft_row(row: dict[str, Any], result: dict[str, Any], teacher_model: str, backend: str) -> dict[str, Any]:
    if not result["exact_correct"] or result["predicted"] is None:
        raise ValueError("Only exact teacher results can become SFT rows.")
    return {
        "id": row["id"],
        "source_index": row["source_index"],
        "source_id": row["source_id"],
        "partition": row["partition"],
        "teacher_model": teacher_model,
        "teacher_backend": backend,
        "input": row["input"],
        "tools": row["tools"],
        "expected_calls": row["expected_calls"],
        "gold_answer": row["gold_answer"],
        "messages": messages_for_row(row, result["predicted"]),
        "teacher_raw_output": result["raw_output"],
    }


def summarize_eval(results: list[dict[str, Any]]) -> EvalSummary:
    total = len(results)
    parsed = sum(1 for result in results if result["predicted"] is not None)
    exact_correct = sum(1 for result in results if result["exact_correct"])
    name_sequence_correct = sum(1 for result in results if result["name_sequence_correct"])
    return EvalSummary(
        total=total,
        parsed=parsed,
        exact_correct=exact_correct,
        name_sequence_correct=name_sequence_correct,
        exact_accuracy=exact_correct / total if total else 0.0,
        name_sequence_accuracy=name_sequence_correct / total if total else 0.0,
        parse_rate=parsed / total if total else 0.0,
    )


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
