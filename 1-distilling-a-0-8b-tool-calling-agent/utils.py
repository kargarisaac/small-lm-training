from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
import ast
import json
import os
import time
import urllib.error
import urllib.request

import bfcl_eval
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
from transformers import AutoModelForCausalLM, AutoTokenizer


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

STUDENT_MODEL = "Qwen/Qwen3.5-0.8B"
TEACHER_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
TOKENIZER_MODEL = STUDENT_MODEL

TRAIN_SIZE = 150
EVAL_SIZE = 50
EVAL_RUN_LIMIT = 50
MAX_STEPS_PER_TURN = 20
MAX_CONSECUTIVE_EXECUTION_ERRORS: int | None = None
MAX_NEW_TOKENS = 2048

BFCL_PACKAGE_DATA_DIR = Path(bfcl_eval.__file__).resolve().parent / "data"
BFCL_V4_QUESTION_FILE = BFCL_PACKAGE_DATA_DIR / "BFCL_v4_multi_turn_base.json"
BFCL_V4_ANSWER_FILE = BFCL_PACKAGE_DATA_DIR / "possible_answer" / "BFCL_v4_multi_turn_base.json"

CLASS_TO_FUNC_DOC = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "TwitterAPI": "posting_api.json",
    "VehicleControlAPI": "vehicle_control.json",
}

GENERATION_STOP_STRINGS = ["<|im_end|>", "<|endoftext|>"]

TEACHER_ACTION_POLICY_SYSTEM_MESSAGE = """You are an action policy agent inside a tool-use benchmark harness.
Your job is to emit the next executable action, not to chat with the user.
If an available tool can advance the task, reply only with the needed <tool_call> block or blocks.
Do not include natural-language explanation before or after tool calls.
Use the tool schema and current conversation state to choose arguments.
Only answer in natural language when the task is complete or when no available tool can help."""


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class BFCLSplit:
    train_entries: list[dict[str, Any]]
    train_answers: list[dict[str, Any]]
    eval_entries: list[dict[str, Any]]
    eval_answers: list[dict[str, Any]]
    benchmark_entries: list[dict[str, Any]]
    benchmark_answers: list[dict[str, Any]]
    all_entries: list[dict[str, Any]]
    all_answers: list[dict[str, Any]]


@dataclass
class ParsedQwenToolCall:
    name: str
    arguments: dict[str, str]
    raw_block: str


@dataclass
class QwenParseResult:
    calls: list[ParsedQwenToolCall]
    errors: list[str]


@dataclass
class TeacherConfig:
    provider: str = os.getenv("TEACHER_PROVIDER", "mlx_raw_server")
    server_base_url: str = os.getenv("TEACHER_SERVER_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    model_name: str = os.getenv("TEACHER_MODEL", TEACHER_MODEL)
    request_model: str = os.getenv("TEACHER_REQUEST_MODEL", "default_model")
    enable_thinking: bool = False
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 20
    request_timeout_seconds: int = 180
    max_new_tokens: int = MAX_NEW_TOKENS


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool = env_flag("BFCL_MLFLOW_ENABLED", False)
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "distillation-blogs-bfcl")


@dataclass(frozen=True)
class LocalTraceConfig:
    enabled: bool = env_flag("BFCL_LOCAL_TRACE_ENABLED", False)
    base_dir: Path = OUTPUT_DIR / "local_traces"


@dataclass(frozen=True)
class GenerationAttemptConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    seed: int
    do_sample: bool = True


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(make_json_safe(row)) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_json_safe(row), ensure_ascii=False) + "\n")


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return repr(value)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_slug(model_name: str) -> str:
    return (
        model_name.replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def load_bfcl_split(
    train_size: int = TRAIN_SIZE,
    eval_size: int = EVAL_SIZE,
    eval_run_limit: int = EVAL_RUN_LIMIT,
) -> BFCLSplit:
    all_entries = load_jsonl(BFCL_V4_QUESTION_FILE)
    all_answers = load_jsonl(BFCL_V4_ANSWER_FILE)

    assert len(all_entries) == len(all_answers)
    for entry, answer in zip(all_entries, all_answers):
        assert entry["id"] == answer["id"], (entry["id"], answer["id"])

    if train_size + eval_size > len(all_entries):
        raise ValueError("train_size + eval_size is larger than the benchmark.")

    train_entries = all_entries[:train_size]
    train_answers = all_answers[:train_size]
    eval_entries = all_entries[train_size:train_size + eval_size]
    eval_answers = all_answers[train_size:train_size + eval_size]
    benchmark_entries = eval_entries[:eval_run_limit]
    benchmark_answers = eval_answers[:eval_run_limit]

    return BFCLSplit(
        train_entries=train_entries,
        train_answers=train_answers,
        eval_entries=eval_entries,
        eval_answers=eval_answers,
        benchmark_entries=benchmark_entries,
        benchmark_answers=benchmark_answers,
        all_entries=all_entries,
        all_answers=all_answers,
    )


def load_package_function_docs(class_name: str) -> list[dict[str, Any]]:
    file_name = CLASS_TO_FUNC_DOC[class_name]
    path = BFCL_PACKAGE_DATA_DIR / "multi_turn_func_doc" / file_name
    tools = load_jsonl(path)
    for tool in tools:
        tool["bfcl_class"] = class_name
    return tools


def load_package_tools_for_example(example: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for class_name in example["involved_classes"]:
        tools.extend(load_package_function_docs(class_name))
    return tools


def build_tool_schema_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schemas = {tool["name"]: tool for tool in tools}
    if len(schemas) != len(tools):
        raise ValueError("Duplicate tool names in the active BFCL tool set.")
    return schemas


def tool_for_qwen_prompt(tool: dict[str, Any]) -> dict[str, Any]:
    clean_tool = {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"],
    }
    if "response" in tool:
        clean_tool["response"] = tool["response"]
    return clean_tool


def tool_for_chat_template(tool: dict[str, Any]) -> dict[str, Any]:
    clean_tool = tool_for_qwen_prompt(tool)
    parameters = json.loads(json.dumps(clean_tool["parameters"]))
    if parameters.get("type") == "dict":
        parameters["type"] = "object"
    clean_tool["parameters"] = parameters
    return clean_tool


def load_tokenizer(model_name: str = TOKENIZER_MODEL):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_student_model_and_tokenizer(model_name: str = STUDENT_MODEL, device: str | None = None):
    import torch

    tokenizer = load_tokenizer(model_name)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generation_eos_token_ids(tokenizer) -> list[int]:
    token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        token_ids.append(im_end_id)
    return token_ids


def make_student_text_generator(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int | None = None,
) -> Callable[[str], str]:
    import torch

    eos_token_ids = generation_eos_token_ids(tokenizer)
    if seed is not None:
        torch.manual_seed(seed)

    def generate_student_text(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": eos_token_ids,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                **generation_kwargs,
            )

        new_token_ids = generated[0, inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_token_ids, skip_special_tokens=False)

    generate_student_text.generation_config = {
        "provider": "local_transformers",
        "model_name": getattr(model, "name_or_path", STUDENT_MODEL),
        "device": device,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    return generate_student_text


def make_student_text_generator_factory(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Callable[[GenerationAttemptConfig], Callable[[str], str]]:
    def make_for_attempt(attempt: GenerationAttemptConfig) -> Callable[[str], str]:
        return make_student_text_generator(
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
            do_sample=attempt.do_sample,
            temperature=attempt.temperature,
            top_p=attempt.top_p,
            top_k=attempt.top_k,
            seed=attempt.seed,
        )

    return make_for_attempt


def adaptive_temperature_retry_attempts_for_task(
    task_index: int,
    entry: dict[str, Any] | None = None,
    temperatures: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    top_p: float = 0.95,
    top_k: int = 20,
    seed_base: int = 1000,
) -> list[GenerationAttemptConfig]:
    return [
        GenerationAttemptConfig(
            name=f"temp_{temperature:g}",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed_base + task_index * 100 + attempt_index,
        )
        for attempt_index, temperature in enumerate(temperatures, start=1)
    ]


def attempt_config_to_dict(attempt: GenerationAttemptConfig | None) -> dict[str, Any] | None:
    if attempt is None:
        return None
    return {
        "name": attempt.name,
        "temperature": attempt.temperature,
        "top_p": attempt.top_p,
        "top_k": attempt.top_k,
        "seed": attempt.seed,
        "do_sample": attempt.do_sample,
    }


def teacher_config_to_dict(config: TeacherConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "server_base_url": config.server_base_url,
        "model_name": config.model_name,
        "request_model": config.request_model,
        "enable_thinking": config.enable_thinking,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "request_timeout_seconds": config.request_timeout_seconds,
        "max_new_tokens": config.max_new_tokens,
    }


def parsed_tool_call_to_dict(call: ParsedQwenToolCall) -> dict[str, Any]:
    return {
        "name": call.name,
        "arguments": call.arguments,
        "raw_block": call.raw_block,
    }


def extract_tag_blocks(text: str, start_tag: str, end_tag: str) -> tuple[list[str], list[str]]:
    blocks: list[str] = []
    errors: list[str] = []
    cursor = 0

    while True:
        start = text.find(start_tag, cursor)
        if start == -1:
            break

        content_start = start + len(start_tag)
        end = text.find(end_tag, content_start)
        if end == -1:
            errors.append(f"Found {start_tag} without matching {end_tag}.")
            break

        blocks.append(text[content_start:end].strip())
        cursor = end + len(end_tag)

    return blocks, errors


def parse_qwen_opening_tag(line: str, prefix: str, label: str) -> tuple[str | None, str, str | None]:
    if not line.startswith(prefix):
        return None, "", f"Expected <{label}=name>, got: {line}"

    remainder = line[len(prefix):]
    separator_index = remainder.find(">")
    if separator_index == -1:
        return None, "", f"Expected <{label}=name>, got: {line}"

    name = remainder[:separator_index].strip()
    inline_value = remainder[separator_index + 1:]
    if not name:
        return None, inline_value, f"{label.capitalize()} name is empty."
    if any(character in name for character in "<>/"):
        return None, inline_value, f"Malformed {label} name: {name!r}"

    return name, inline_value, None


def split_inline_parameter_close(value: str) -> tuple[str, bool, str]:
    close_tag = "</parameter>"
    if close_tag not in value:
        return value, False, ""

    parameter_value, trailing = value.split(close_tag, 1)
    return parameter_value, True, trailing.strip()


def parse_qwen_tool_call_block(block: str) -> tuple[ParsedQwenToolCall | None, list[str]]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    errors: list[str] = []

    if not lines:
        return None, ["Empty tool_call block."]

    function_line = lines[0]
    function_name, function_inline_value, function_error = parse_qwen_opening_tag(
        function_line,
        "<function=",
        "function",
    )
    if function_error is not None:
        return None, [function_error]
    if function_inline_value:
        errors.append(f"Unexpected content after <function={function_name}>: {function_inline_value}")

    arguments: dict[str, str] = {}
    index = 1
    while index < len(lines):
        line = lines[index]

        if line == "</function>":
            trailing = lines[index + 1:]
            if trailing:
                errors.append(f"Unexpected content after </function>: {trailing}")
            break

        parameter_name, inline_value, parameter_error = parse_qwen_opening_tag(
            line,
            "<parameter=",
            "parameter",
        )
        if parameter_error is not None:
            errors.append(parameter_error)
            index += 1
            continue

        value_lines: list[str] = []
        inline_value, parameter_closed_inline, trailing_after_close = split_inline_parameter_close(inline_value)
        if inline_value:
            value_lines.append(inline_value)
        if trailing_after_close:
            errors.append(f"Unexpected content after </parameter>: {trailing_after_close}")

        index += 1
        if parameter_closed_inline:
            if parameter_name:
                arguments[parameter_name] = "\n".join(value_lines).strip()
            continue

        while index < len(lines) and lines[index] != "</parameter>":
            value_lines.append(lines[index])
            index += 1

        if index >= len(lines):
            errors.append(f"Parameter {parameter_name} is missing </parameter>.")
            break

        if parameter_name:
            arguments[parameter_name] = "\n".join(value_lines).strip()

        index += 1
    else:
        errors.append("tool_call block is missing </function>.")

    if errors:
        return None, errors

    return ParsedQwenToolCall(name=function_name or "", arguments=arguments, raw_block=block), []


def parse_qwen_tool_calls(text: str) -> QwenParseResult:
    text = text.split("<|im_end|>", 1)[0]
    blocks, errors = extract_tag_blocks(text, "<tool_call>", "</tool_call>")
    calls: list[ParsedQwenToolCall] = []

    for block in blocks:
        call, block_errors = parse_qwen_tool_call_block(block)
        errors.extend(block_errors)
        if call is not None:
            calls.append(call)

    if not blocks:
        errors.append("No <tool_call> block found.")

    return QwenParseResult(calls=calls, errors=errors)


def coerce_argument_value(value: str, schema: dict[str, Any]) -> Any:
    target_type = schema.get("type", "string")

    if target_type == "integer":
        return int(value)
    if target_type == "float":
        return float(value)
    if target_type == "boolean":
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Cannot coerce {value!r} to boolean.")
    if target_type in {"array", "object", "dict"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def qwen_call_to_bfcl_execution_string_for_tools(
    call: ParsedQwenToolCall,
    schemas: dict[str, dict[str, Any]],
) -> str:
    if call.name not in schemas:
        raise ValueError(f"Unknown tool name: {call.name}")

    tool_schema = schemas[call.name]
    parameter_schemas = tool_schema.get("parameters", {}).get("properties", {})
    required_parameters = set(tool_schema.get("parameters", {}).get("required", []))
    provided_parameters = set(call.arguments.keys())

    missing = sorted(required_parameters - provided_parameters)
    if missing:
        raise ValueError(f"Missing required parameters for {call.name}: {missing}")

    coerced_arguments: dict[str, Any] = {}
    for name, value in call.arguments.items():
        if name not in parameter_schemas:
            raise ValueError(f"Unknown parameter for {call.name}: {name}")
        coerced_arguments[name] = coerce_argument_value(value, parameter_schemas[name])

    argument_text = ",".join(f"{name}={repr(value)}" for name, value in coerced_arguments.items())
    return f"{call.name}({argument_text})"


def strip_generated_special_tokens(text: str) -> str:
    cleaned = text.split("<|im_end|>", 1)[0].strip()
    for special_token in ["<|endoftext|>"]:
        cleaned = cleaned.replace(special_token, "")
    return cleaned.strip()


def render_tool_response_messages(execution_results: list[str]) -> list[dict[str, str]]:
    return [{"role": "tool", "content": result} for result in execution_results]


def generation_config_for_callable(generate_callable: Callable[..., Any] | None) -> dict[str, Any] | None:
    if generate_callable is None:
        return None
    config = getattr(generate_callable, "generation_config", None)
    return make_json_safe(config) if isinstance(config, dict) else None


def endpoint_call_for_callable(generate_callable: Callable[..., Any] | None) -> dict[str, Any] | None:
    if generate_callable is None:
        return None
    call = getattr(generate_callable, "last_endpoint_call", None)
    return make_json_safe(call) if isinstance(call, dict) else None


def is_execution_error(result: str) -> bool:
    return result.startswith("Error during execution:") or '"error"' in result


def run_bfcl_entry_harness(
    entry: dict[str, Any],
    run_name: str,
    tokenizer,
    generate_text: Callable[[str], str] | None = None,
    generate_action: Callable[..., str] | None = None,
    capture_prompts: bool = False,
    max_steps_per_turn: int = MAX_STEPS_PER_TURN,
    max_consecutive_execution_errors: int | None = MAX_CONSECUTIVE_EXECUTION_ERRORS,
    prompt_messages_prefix: list[dict[str, str]] | None = None,
    local_trace_logger: LocalTraceRunLogger | None = None,
) -> dict[str, Any]:
    if generate_text is None and generate_action is None:
        raise ValueError("Provide either generate_text or generate_action.")

    tools = load_package_tools_for_example(entry)
    chat_tools = [tool_for_chat_template(tool) for tool in tools]
    schemas = build_tool_schema_map(tools)

    messages: list[dict[str, str]] = [dict(message) for message in (prompt_messages_prefix or [])]
    raw_turns: list[list[str]] = []
    decoded_turns: list[list[list[str]]] = []
    trace_rows: list[dict[str, Any]] = []
    trace_run_id = f"{run_name}_{entry['id']}_{uuid4().hex[:8]}"
    long_context = "long_context" in entry["id"] or "composite" in entry["id"]

    if local_trace_logger is not None:
        local_trace_logger.append_event(
            "task_harness_started",
            task_id=entry["id"],
            trace_run_id=trace_run_id,
            involved_classes=entry["involved_classes"],
            user_turn_count=len(entry["question"]),
            tool_count=len(chat_tools),
            tools=chat_tools,
            initial_config=entry.get("initial_config"),
        )

    for turn_index, user_messages in enumerate(entry["question"]):
        messages.extend(dict(message) for message in user_messages)
        raw_steps: list[str] = []
        decoded_steps: list[list[str]] = []
        consecutive_errors = 0

        for step_index in range(max_steps_per_turn):
            messages_before_action = [dict(message) for message in messages]
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=chat_tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            llm_request = {
                "messages": messages_before_action,
                "tools": chat_tools,
                "rendered_prompt": prompt,
                "generation_config": generation_config_for_callable(generate_action or generate_text),
            }

            try:
                if generate_action is None:
                    assert generate_text is not None
                    raw_output = generate_text(prompt)
                else:
                    raw_output = generate_action(
                        messages=messages_before_action,
                        tools=chat_tools,
                        prompt=prompt,
                    )
            except Exception as error:
                if local_trace_logger is not None:
                    local_trace_logger.append_event(
                        "generation_error",
                        task_id=entry["id"],
                        turn_index=turn_index,
                        step_index=step_index,
                        error_type=type(error).__name__,
                        error_message=str(error),
                        llm_request=llm_request,
                        endpoint_call=endpoint_call_for_callable(generate_action),
                    )
                raise

            endpoint_call = endpoint_call_for_callable(generate_action)
            llm_response = {
                "raw_output": raw_output,
                "endpoint_call": endpoint_call,
            }

            def log_harness_step(
                trace_row: dict[str, Any],
                parse_result: QwenParseResult,
                execution_calls: list[str] | None = None,
                execution_results: list[str] | None = None,
            ) -> None:
                if local_trace_logger is None:
                    return
                local_trace_logger.append_harness_step(
                    task_id=entry["id"],
                    turn_index=turn_index,
                    step_index=step_index,
                    stop_reason=str(trace_row.get("stop_reason")),
                    llm_request=llm_request,
                    llm_response={
                        **llm_response,
                        "assistant_message_content": cleaned_output,
                    },
                    parsed={
                        "calls": [parsed_tool_call_to_dict(call) for call in parse_result.calls],
                        "errors": parse_result.errors,
                    },
                    execution={
                        "calls": execution_calls or [],
                        "results": execution_results or [],
                    },
                    trace_row=trace_row,
                )
            raw_steps.append(raw_output)

            parse_result = parse_qwen_tool_calls(raw_output)
            cleaned_output = strip_generated_special_tokens(raw_output)
            messages.append({"role": "assistant", "content": cleaned_output})

            if not parse_result.calls:
                trace_row = {
                    "turn": turn_index,
                    "step": step_index,
                    "stop_reason": "no_tool_call",
                    "raw_output": raw_output,
                    "parse_errors": parse_result.errors,
                }
                if capture_prompts:
                    trace_row["prompt"] = prompt
                    trace_row["messages_before_action"] = messages_before_action
                    trace_row["assistant_message_content"] = cleaned_output
                trace_rows.append(trace_row)
                log_harness_step(trace_row, parse_result)
                break

            try:
                execution_calls = [
                    qwen_call_to_bfcl_execution_string_for_tools(call, schemas)
                    for call in parse_result.calls
                ]
            except ValueError as error:
                trace_row = {
                    "turn": turn_index,
                    "step": step_index,
                    "stop_reason": "invalid_tool_call",
                    "raw_output": raw_output,
                    "parse_errors": parse_result.errors,
                    "error": str(error),
                }
                if capture_prompts:
                    trace_row["prompt"] = prompt
                    trace_row["messages_before_action"] = messages_before_action
                    trace_row["assistant_message_content"] = cleaned_output
                trace_rows.append(trace_row)
                log_harness_step(trace_row, parse_result)
                break

            decoded_steps.append(execution_calls)
            execution_results, _ = execute_multi_turn_func_call(
                func_call_list=execution_calls,
                initial_config=entry["initial_config"],
                involved_classes=entry["involved_classes"],
                model_name=trace_run_id,
                test_entry_id=entry["id"],
                long_context=long_context,
                is_evaL_run=False,
            )
            messages.extend(render_tool_response_messages(execution_results))

            if any(is_execution_error(result) for result in execution_results):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            trace_row = {
                "turn": turn_index,
                "step": step_index,
                "stop_reason": "executed_tool_call",
                "raw_output": raw_output,
                "execution_calls": execution_calls,
                "execution_results": execution_results,
                "parse_errors": parse_result.errors,
            }
            if capture_prompts:
                trace_row["prompt"] = prompt
                trace_row["messages_before_action"] = messages_before_action
                trace_row["assistant_message_content"] = cleaned_output
            trace_rows.append(trace_row)
            log_harness_step(trace_row, parse_result, execution_calls, execution_results)

            if (
                max_consecutive_execution_errors is not None
                and consecutive_errors >= max_consecutive_execution_errors
            ):
                trace_rows[-1]["stop_reason"] = "too_many_execution_errors"
                break
        else:
            trace_row = {
                "turn": turn_index,
                "step": max_steps_per_turn,
                "stop_reason": "max_steps_per_turn",
            }
            trace_rows.append(trace_row)
            if local_trace_logger is not None:
                local_trace_logger.append_event(
                    "turn_stopped",
                    task_id=entry["id"],
                    turn_index=turn_index,
                    step_index=max_steps_per_turn,
                    stop_reason="max_steps_per_turn",
                    trace_row=trace_row,
                )

        raw_turns.append(raw_steps)
        decoded_turns.append(decoded_steps)

    return {
        "id": entry["id"],
        "raw_turns": raw_turns,
        "decoded_turns": decoded_turns,
        "trace": trace_rows,
    }


def score_bfcl_trace(
    entry: dict[str, Any],
    answer: dict[str, Any],
    decoded_turns: list[list[list[str]]],
    run_name: str,
) -> dict[str, Any]:
    if len(decoded_turns) != len(answer["ground_truth"]):
        return {
            "valid": False,
            "error_type": "notebook:wrong_turn_count",
            "error_message": (
                f"Model produced {len(decoded_turns)} turns, "
                f"but ground truth has {len(answer['ground_truth'])}."
            ),
        }

    return multi_turn_checker(
        multi_turn_model_result_list_decoded=decoded_turns,
        multi_turn_ground_truth_list=answer["ground_truth"],
        test_entry=entry,
        test_category="multi_turn_base",
        model_name=run_name,
    )


def summarize_score(score: dict[str, Any]) -> str:
    if score.get("valid"):
        return "valid"
    return score.get("error_type", "unknown_error")


def runtime_error_trace(entry: dict[str, Any], error_message: str) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "raw_turns": [],
        "decoded_turns": [],
        "trace": [
            {
                "turn": None,
                "step": None,
                "stop_reason": "runtime_error",
                "raw_output": "",
                "parse_errors": [],
                "error_message": error_message,
            }
        ],
    }


def is_fatal_generation_error(error: Exception) -> bool:
    message = str(error)
    return (
        message.startswith("Could not reach the MLX teacher server")
        or message.startswith("MLX teacher HTTP")
    )


def run_bfcl_entry_harness_with_retry(
    entry: dict[str, Any],
    answer: dict[str, Any],
    run_name: str,
    tokenizer,
    attempt_configs: list[GenerationAttemptConfig],
    generate_text_factory: Callable[[GenerationAttemptConfig], Callable[[str], str]] | None = None,
    generate_action_factory: Callable[[GenerationAttemptConfig], Callable[..., str]] | None = None,
    capture_prompts: bool = False,
    max_steps_per_turn: int = MAX_STEPS_PER_TURN,
    max_consecutive_execution_errors: int | None = MAX_CONSECUTIVE_EXECUTION_ERRORS,
    prompt_messages_prefix: list[dict[str, str]] | None = None,
    local_trace_logger: LocalTraceRunLogger | None = None,
) -> dict[str, Any]:
    if generate_text_factory is None and generate_action_factory is None:
        raise ValueError("Provide either generate_text_factory or generate_action_factory.")
    if not attempt_configs:
        return {
            "id": entry["id"],
            "valid": False,
            "score": {"valid": False, "error_type": "no_attempts"},
            "trace": {"id": entry["id"], "raw_turns": [], "decoded_turns": [], "trace": []},
            "attempts": [],
            "attempt_traces": [],
            "selected_attempt": None,
        }

    attempts: list[dict[str, Any]] = []
    selected_trace: dict[str, Any] | None = None
    selected_score: dict[str, Any] | None = None
    selected_attempt: dict[str, Any] | None = None
    attempt_traces: list[dict[str, Any]] = []

    for attempt_index, attempt_config in enumerate(attempt_configs, start=1):
        started_at = time.time()
        attempt_run_name = f"{run_name}_attempt_{attempt_index}_{attempt_config.name}"
        if local_trace_logger is not None:
            local_trace_logger.append_event(
                "attempt_started",
                task_id=entry["id"],
                attempt_index=attempt_index,
                attempt_config=attempt_config_to_dict(attempt_config),
                attempt_run_name=attempt_run_name,
            )

        try:
            generate_text = (
                generate_text_factory(attempt_config)
                if generate_text_factory is not None
                else None
            )
            generate_action = (
                generate_action_factory(attempt_config)
                if generate_action_factory is not None
                else None
            )
            trace = run_bfcl_entry_harness(
                entry,
                run_name=attempt_run_name,
                tokenizer=tokenizer,
                generate_text=generate_text,
                generate_action=generate_action,
                capture_prompts=capture_prompts,
                prompt_messages_prefix=prompt_messages_prefix,
                max_steps_per_turn=max_steps_per_turn,
                max_consecutive_execution_errors=max_consecutive_execution_errors,
                local_trace_logger=local_trace_logger,
            )
            score = score_bfcl_trace(entry, answer, trace["decoded_turns"], run_name=attempt_run_name)
            error_message = None
        except Exception as error:
            trace = None
            score = {
                "valid": False,
                "error_type": "runtime_error",
                "error_message": str(error),
            }
            error_message = str(error)

        elapsed_seconds = round(time.time() - started_at, 2)
        attempt_summary = {
            "attempt_index": attempt_index,
            "attempt_config": attempt_config_to_dict(attempt_config),
            "valid": bool(score.get("valid")),
            "score": make_json_safe(score),
            "trace_summary": summarize_trace_for_attempt(trace),
            "elapsed_seconds": elapsed_seconds,
            "error_message": error_message,
        }
        attempts.append(attempt_summary)
        attempt_traces.append(
            {
                "attempt_index": attempt_index,
                "attempt_config": attempt_config_to_dict(attempt_config),
                "valid": bool(score.get("valid")),
                "score": make_json_safe(score),
                "trace": make_json_safe(trace) if trace is not None else None,
                "elapsed_seconds": elapsed_seconds,
                "error_message": error_message,
            }
        )
        if local_trace_logger is not None:
            local_trace_logger.append_event(
                "attempt_finished",
                task_id=entry["id"],
                attempt_index=attempt_index,
                attempt_config=attempt_config_to_dict(attempt_config),
                valid=bool(score.get("valid")),
                score=make_json_safe(score),
                elapsed_seconds=elapsed_seconds,
                error_message=error_message,
            )

        if score.get("valid") and trace is not None:
            selected_trace = trace
            selected_score = score
            selected_attempt = attempt_summary
            break

    if selected_trace is None:
        selected_trace = trace or {"id": entry["id"], "raw_turns": [], "decoded_turns": [], "trace": []}
        selected_score = score
        selected_attempt = attempts[-1] if attempts else None

    return {
        "id": entry["id"],
        "valid": bool(selected_score and selected_score.get("valid")),
        "score": selected_score or {"valid": False, "error_type": "no_attempts"},
        "trace": selected_trace,
        "attempts": attempts,
        "attempt_traces": attempt_traces,
        "selected_attempt": selected_attempt,
    }


def load_evaluation_results_cache(
    output_path: Path | None,
    requested_ids: set[str],
) -> tuple[list[dict[str, Any]], str | None]:
    if output_path is None or not output_path.exists():
        return [], None

    payload = load_json_file(output_path, default={})
    if isinstance(payload, list):
        cached_results = payload
        cached_run_name = None
    elif isinstance(payload, dict):
        cached_results = payload.get("results", [])
        cached_run_name = payload.get("run_name")
    else:
        raise ValueError(f"Unsupported eval cache shape in {output_path}: {type(payload).__name__}")

    filtered_results = [
        result
        for result in cached_results
        if isinstance(result, dict) and result.get("id") in requested_ids
    ]
    return filtered_results, cached_run_name


def summarize_evaluation_results(
    *,
    run_name: str,
    results: list[dict[str, Any]],
    uses_retry: bool,
    requested_count: int,
) -> dict[str, Any]:
    correct_count = sum(1 for result in results if result.get("valid"))
    if uses_retry:
        first_attempt_correct_count = sum(
            1
            for result in results
            if result.get("attempts") and result["attempts"][0].get("valid")
        )
    else:
        first_attempt_correct_count = correct_count

    completed_count = len(results)
    return {
        "run_name": run_name,
        "accuracy": correct_count / completed_count if completed_count else 0.0,
        "correct_count": correct_count,
        "total_count": completed_count,
        "requested_count": requested_count,
        "first_attempt_correct_count": first_attempt_correct_count,
        "uses_retry": uses_retry,
        "results": results,
    }


def write_evaluation_cache(output_path: Path | None, summary: dict[str, Any]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_json_safe(summary), indent=2), encoding="utf-8")


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def setup_mlflow(config: MlflowConfig):
    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    return mlflow


def score_error_type(score: dict[str, Any]) -> str:
    if score.get("valid"):
        return "valid"
    return str(score.get("error_type") or "unknown")


def trace_metrics_from_rows(trace_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "trace_rows": len(trace_rows),
        "executed_tool_call_rows": sum(
            1 for row in trace_rows if row.get("stop_reason") == "executed_tool_call"
        ),
        "parse_error_rows": sum(1 for row in trace_rows if row.get("parse_errors")),
        "no_tool_call_rows": sum(1 for row in trace_rows if row.get("stop_reason") == "no_tool_call"),
    }


def log_mlflow_dict_artifact(mlflow, payload: dict[str, Any], artifact_file: str) -> None:
    mlflow.log_dict(make_json_safe(payload), artifact_file)


def log_bfcl_result_as_mlflow_trace(
    mlflow,
    result: dict[str, Any],
    *,
    trace_name: str,
    kind: str,
    tags: dict[str, str] | None = None,
) -> None:
    score = result.get("score", {})
    trace_rows = result.get("trace", [])
    if isinstance(trace_rows, dict):
        trace_rows = trace_rows.get("trace", [])
    if not isinstance(trace_rows, list):
        trace_rows = []

    attributes = {
        "bfcl.kind": kind,
        "bfcl.task_id": str(result.get("id", "")),
        "bfcl.valid": bool(result.get("valid")),
        "bfcl.error_type": score_error_type(score) if isinstance(score, dict) else "unknown",
        "trace_rows": len(trace_rows),
        **(tags or {}),
    }

    with mlflow.start_span(trace_name, span_type="CHAIN", attributes=make_json_safe(attributes)) as root_span:
        for key, value in attributes.items():
            mlflow.set_trace_tag(root_span.trace_id, key, str(value)[:250])

        root_span.set_inputs(
            make_json_safe(
                {
                    "id": result.get("id"),
                    "attempts": result.get("attempts", []),
                    "selected_attempt": result.get("selected_attempt"),
                }
            )
        )

        for row in trace_rows:
            if not isinstance(row, dict):
                continue
            span_name = f"turn_{row.get('turn', 'x')}_step_{row.get('step', 'x')}_{row.get('stop_reason', 'unknown')}"
            span_type = "TOOL" if row.get("stop_reason") == "executed_tool_call" else "PARSER"
            with mlflow.start_span(
                span_name,
                span_type=span_type,
                attributes=make_json_safe(
                    {
                        "turn": row.get("turn"),
                        "step": row.get("step"),
                        "stop_reason": row.get("stop_reason"),
                        "parse_error_count": len(row.get("parse_errors", []) or []),
                    }
                ),
            ) as step_span:
                step_span.set_inputs(
                    make_json_safe(
                        {
                            "prompt": row.get("prompt"),
                            "messages_before_action": row.get("messages_before_action"),
                        }
                    )
                )
                step_span.set_outputs(
                    make_json_safe(
                        {
                            "raw_output": row.get("raw_output"),
                            "assistant_message_content": row.get("assistant_message_content"),
                            "parse_errors": row.get("parse_errors", []),
                            "execution_calls": row.get("execution_calls", []),
                            "execution_results": row.get("execution_results", []),
                            "error": row.get("error"),
                        }
                    )
                )

        root_span.set_outputs(
            make_json_safe(
                {
                    "valid": bool(result.get("valid")),
                    "score": score,
                    "decoded_turns": result.get("decoded_turns", []),
                }
            )
        )


def safe_artifact_name(text: str) -> str:
    characters = [
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in text
    ]
    return "".join(characters).strip("._") or "artifact"


class LocalTraceRunLogger:
    schema_version = 1

    def __init__(
        self,
        *,
        run_name: str,
        kind: str,
        config: LocalTraceConfig,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.run_name = run_name
        self.kind = kind
        self.tags = tags or {}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = config.base_dir / f"{timestamp}_{safe_artifact_name(run_name)}"
        self.tasks_dir = self.run_dir / "tasks"
        self.events_path = self.run_dir / "events.jsonl"
        self.scoreboard_path = self.run_dir / "scoreboard.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.append_event(
            "run_started",
            kind=kind,
            tags=self.tags,
            local_trace_dir=str(self.run_dir),
        )

    def append_event(self, event_type: str, **payload: Any) -> None:
        append_jsonl_row(
            self.events_path,
            {
                "schema_version": self.schema_version,
                "event_id": uuid4().hex,
                "timestamp": utc_now_iso(),
                "run_name": self.run_name,
                "kind": self.kind,
                "event_type": event_type,
                **payload,
            },
        )

    def append_harness_step(
        self,
        *,
        task_id: str,
        turn_index: int,
        step_index: int,
        stop_reason: str,
        llm_request: dict[str, Any],
        llm_response: dict[str, Any],
        parsed: dict[str, Any],
        execution: dict[str, Any],
        trace_row: dict[str, Any],
    ) -> None:
        self.append_event(
            "harness_step",
            task_id=task_id,
            turn_index=turn_index,
            step_index=step_index,
            stop_reason=stop_reason,
            llm_request=llm_request,
            llm_response=llm_response,
            parsed=parsed,
            execution=execution,
            trace_row=trace_row,
        )

    def write_task_result(self, result: dict[str, Any]) -> None:
        task_id = safe_artifact_name(str(result.get("id", "unknown_task")))
        task_path = self.tasks_dir / f"{task_id}.json"
        task_path.write_text(json.dumps(make_json_safe(result), indent=2), encoding="utf-8")

        score = result.get("score", {})
        append_jsonl_row(
            self.scoreboard_path,
            {
                "timestamp": utc_now_iso(),
                "id": result.get("id"),
                "valid": bool(result.get("valid")),
                "error_type": score_error_type(score) if isinstance(score, dict) else "unknown",
                "trace_rows": len(result.get("trace", [])) if isinstance(result.get("trace"), list) else 0,
                "attempt_count": len(result.get("attempts", [])) if isinstance(result.get("attempts"), list) else 0,
                "task_path": str(task_path),
            },
        )
        self.append_event(
            "task_result_written",
            task_id=result.get("id"),
            valid=bool(result.get("valid")),
            error_type=score_error_type(score) if isinstance(score, dict) else "unknown",
            task_path=str(task_path),
        )

    def write_summary(self, summary: dict[str, Any]) -> None:
        summary_with_trace_path = dict(summary)
        summary_with_trace_path["local_trace_dir"] = str(self.run_dir)
        self.summary_path.write_text(
            json.dumps(make_json_safe(summary_with_trace_path), indent=2),
            encoding="utf-8",
        )
        self.append_event(
            "summary_written",
            accuracy=summary.get("accuracy"),
            correct_count=summary.get("correct_count"),
            total_count=summary.get("total_count"),
            requested_count=summary.get("requested_count"),
            summary_path=str(self.summary_path),
        )


def create_local_trace_logger(
    config: LocalTraceConfig | None,
    *,
    run_name: str,
    kind: str,
    tags: dict[str, str] | None = None,
) -> LocalTraceRunLogger | None:
    if config is None or not config.enabled:
        return None
    return LocalTraceRunLogger(run_name=run_name, kind=kind, config=config, tags=tags)


def write_evaluation_trace_bundle(summary: dict[str, Any]) -> Path:
    run_name = safe_artifact_name(str(summary.get("run_name", "bfcl_eval")))
    bundle_dir = OUTPUT_DIR / "mlflow_trace_exports" / run_name
    tasks_dir = bundle_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(json.dumps(make_json_safe(summary), indent=2), encoding="utf-8")

    scoreboard_rows: list[dict[str, Any]] = []
    for result in summary.get("results", []):
        if not isinstance(result, dict):
            continue

        task_id = safe_artifact_name(str(result.get("id", "unknown_task")))
        (tasks_dir / f"{task_id}.json").write_text(
            json.dumps(make_json_safe(result), indent=2),
            encoding="utf-8",
        )

        trace_rows = result.get("trace", [])
        attempts = result.get("attempts", [])
        score = result.get("score", {})
        scoreboard_rows.append(
            {
                "id": result.get("id"),
                "valid": bool(result.get("valid")),
                "error_type": score_error_type(score) if isinstance(score, dict) else "unknown",
                "trace_rows": len(trace_rows) if isinstance(trace_rows, list) else 0,
                "attempt_count": len(attempts) if isinstance(attempts, list) else 0,
            }
        )

    scoreboard_path = bundle_dir / "scoreboard.jsonl"
    scoreboard_path.write_text(
        "\n".join(json.dumps(row) for row in scoreboard_rows) + ("\n" if scoreboard_rows else ""),
        encoding="utf-8",
    )
    return bundle_dir


def log_bfcl_task_result_to_mlflow(
    result: dict[str, Any],
    *,
    run_name: str,
    kind: str,
    config: MlflowConfig | None,
    tags: dict[str, str] | None = None,
) -> None:
    if config is None or not config.enabled:
        return

    try:
        mlflow = setup_mlflow(config)
        score = result.get("score", {})
        trace_rows = result.get("trace", [])
        if isinstance(trace_rows, dict):
            trace_rows = trace_rows.get("trace", [])
        if not isinstance(trace_rows, list):
            trace_rows = []

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tags(
                {
                    "bfcl.kind": kind,
                    "bfcl.task_id": str(result.get("id", "")),
                    "bfcl.valid": str(bool(result.get("valid"))),
                    "bfcl.error_type": score_error_type(score) if isinstance(score, dict) else "unknown",
                    **(tags or {}),
                }
            )
            mlflow.log_metric("valid", 1 if result.get("valid") else 0)
            for metric_name, metric_value in trace_metrics_from_rows(trace_rows).items():
                mlflow.log_metric(metric_name, metric_value)
            attempts = result.get("attempts", [])
            if isinstance(attempts, list):
                mlflow.log_metric("attempt_count", len(attempts))
            log_mlflow_dict_artifact(mlflow, result, "result.json")
            log_mlflow_dict_artifact(
                mlflow,
                {
                    "id": result.get("id"),
                    "trace": result.get("trace", []),
                    "attempt_traces": result.get("attempt_traces", []),
                },
                "full_trace.json",
            )
            log_bfcl_result_as_mlflow_trace(
                mlflow,
                result,
                trace_name=run_name,
                kind=kind,
                tags=tags,
            )
    except Exception as error:
        print(f"MLflow task logging failed for {result.get('id')}: {error}")


def log_evaluation_summary_to_mlflow(
    summary: dict[str, Any],
    *,
    output_path: Path | None,
    kind: str,
    config: MlflowConfig | None,
    tags: dict[str, str] | None = None,
) -> None:
    if config is None or not config.enabled:
        return

    try:
        mlflow = setup_mlflow(config)
        with mlflow.start_run(run_name=f"{summary['run_name']}_summary", nested=True):
            mlflow.set_tags({"bfcl.kind": kind, **(tags or {})})
            mlflow.log_metric("accuracy", float(summary.get("accuracy", 0.0)))
            mlflow.log_metric("correct_count", int(summary.get("correct_count", 0)))
            mlflow.log_metric("total_count", int(summary.get("total_count", 0)))
            mlflow.log_metric("first_attempt_correct_count", int(summary.get("first_attempt_correct_count", 0)))
            mlflow.log_param("run_name", summary.get("run_name", ""))
            mlflow.log_param("uses_retry", bool(summary.get("uses_retry")))
            log_mlflow_dict_artifact(mlflow, summary, "summary.json")
            trace_bundle_dir = write_evaluation_trace_bundle(summary)
            mlflow.log_artifacts(str(trace_bundle_dir), artifact_path="trace_bundle")
            if output_path is not None and output_path.exists():
                mlflow.log_artifact(str(output_path), artifact_path="outputs")
    except Exception as error:
        print(f"MLflow summary logging failed for {summary.get('run_name')}: {error}")


def log_teacher_collection_to_mlflow(
    summary: dict[str, Any],
    *,
    attempts_output_path: Path | None,
    traces_output_path: Path | None,
    sft_rows_output_path: Path | None,
    config: MlflowConfig | None,
    tags: dict[str, str] | None = None,
) -> None:
    if config is None or not config.enabled:
        return

    try:
        mlflow = setup_mlflow(config)
        with mlflow.start_run(run_name="teacher_train_trajectory_collection", nested=True):
            mlflow.set_tags({"bfcl.kind": "teacher_train_trajectory_collection", **(tags or {})})
            mlflow.log_metric("attempt_count", len(summary.get("attempts", [])))
            mlflow.log_metric("successful_trace_count", len(summary.get("selected_traces", [])))
            mlflow.log_metric("sft_row_count", len(summary.get("sft_rows", [])))
            log_mlflow_dict_artifact(mlflow, summary, "collection_summary.json")
            for path in [attempts_output_path, traces_output_path, sft_rows_output_path]:
                if path is not None and path.exists():
                    mlflow.log_artifact(str(path), artifact_path="outputs")
    except Exception as error:
        print(f"MLflow teacher collection logging failed: {error}")


def evaluate_entries(
    entries: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    tokenizer,
    run_name_prefix: str,
    generate_text: Callable[[str], str] | None = None,
    generate_action: Callable[..., str] | None = None,
    attempt_configs_factory: Callable[[int, dict[str, Any]], list[GenerationAttemptConfig]] | None = None,
    generate_text_factory: Callable[[GenerationAttemptConfig], Callable[[str], str]] | None = None,
    generate_action_factory: Callable[[GenerationAttemptConfig], Callable[..., str]] | None = None,
    output_path: Path | None = None,
    capture_prompts: bool = False,
    prompt_messages_prefix: list[dict[str, str]] | None = None,
    max_steps_per_turn: int = MAX_STEPS_PER_TURN,
    max_consecutive_execution_errors: int | None = MAX_CONSECUTIVE_EXECUTION_ERRORS,
    mlflow_config: MlflowConfig | None = None,
    mlflow_tags: dict[str, str] | None = None,
    local_trace_config: LocalTraceConfig | None = None,
) -> dict[str, Any]:
    requested_ids = {entry["id"] for entry in entries}
    cached_results, cached_run_name = load_evaluation_results_cache(output_path, requested_ids)
    run_name = cached_run_name or f"{run_name_prefix}_{uuid4().hex[:8]}"
    results: list[dict[str, Any]] = list(cached_results)
    completed_ids = {result["id"] for result in results}
    uses_retry = attempt_configs_factory is not None
    local_trace_logger = create_local_trace_logger(
        local_trace_config,
        run_name=run_name,
        kind="bfcl_eval",
        tags=mlflow_tags,
    )

    if cached_results and output_path is not None:
        print(f"Loaded {len(cached_results)} cached eval results from: {output_path}")
    if local_trace_logger is not None:
        print(f"Local trace directory: {local_trace_logger.run_dir}")
        local_trace_logger.append_event(
            "cache_loaded",
            output_path=str(output_path) if output_path is not None else None,
            cached_count=len(cached_results),
            requested_count=len(entries),
        )
        for cached_result in cached_results:
            local_trace_logger.write_task_result(cached_result)
            local_trace_logger.append_event(
                "task_skipped_cached",
                task_id=cached_result.get("id"),
                source_output_path=str(output_path) if output_path is not None else None,
            )

    for task_index, (entry, answer) in enumerate(zip(entries, answers)):
        assert entry["id"] == answer["id"], (entry["id"], answer["id"])
        if entry["id"] in completed_ids:
            print(f"Skipping {entry['id']} (cached)")
            continue

        print(f"Running {entry['id']}...")
        if local_trace_logger is not None:
            local_trace_logger.append_event(
                "task_started",
                task_id=entry["id"],
                task_index=task_index,
                involved_classes=entry["involved_classes"],
            )
        retry_attempts: list[dict[str, Any]] = []
        retry_attempt_traces: list[dict[str, Any]] = []
        selected_attempt: dict[str, Any] | None = None

        try:
            if attempt_configs_factory is None:
                trace = run_bfcl_entry_harness(
                    entry,
                    run_name=run_name,
                    tokenizer=tokenizer,
                    generate_text=generate_text,
                    generate_action=generate_action,
                    capture_prompts=capture_prompts,
                    prompt_messages_prefix=prompt_messages_prefix,
                    max_steps_per_turn=max_steps_per_turn,
                    max_consecutive_execution_errors=max_consecutive_execution_errors,
                    local_trace_logger=local_trace_logger,
                )
                score = score_bfcl_trace(entry, answer, trace["decoded_turns"], run_name=run_name)
            else:
                retry_result = run_bfcl_entry_harness_with_retry(
                    entry,
                    answer,
                    run_name=run_name,
                    tokenizer=tokenizer,
                    attempt_configs=attempt_configs_factory(task_index, entry),
                    generate_text_factory=generate_text_factory,
                    generate_action_factory=generate_action_factory,
                    capture_prompts=capture_prompts,
                    prompt_messages_prefix=prompt_messages_prefix,
                    max_steps_per_turn=max_steps_per_turn,
                    max_consecutive_execution_errors=max_consecutive_execution_errors,
                    local_trace_logger=local_trace_logger,
                )
                trace = retry_result["trace"]
                score = retry_result["score"]
                retry_attempts = retry_result["attempts"]
                retry_attempt_traces = retry_result.get("attempt_traces", [])
                selected_attempt = retry_result["selected_attempt"]
        except Exception as error:
            if is_fatal_generation_error(error):
                raise
            error_message = str(error)
            trace = runtime_error_trace(entry, error_message)
            score = {
                "valid": False,
                "error_type": "runtime_error",
                "error_message": error_message,
            }

        result = {
            "id": entry["id"],
            "valid": bool(score.get("valid")),
            "score": make_json_safe(score),
            "decoded_turns": trace["decoded_turns"],
            "trace": make_json_safe(trace["trace"]),
            "attempts": make_json_safe(retry_attempts),
            "selected_attempt": make_json_safe(selected_attempt),
        }
        if retry_attempt_traces:
            result["attempt_traces"] = make_json_safe(retry_attempt_traces)

        results.append(result)
        print("  score:", summarize_score(score))
        if selected_attempt is not None:
            print("  selected attempt:", selected_attempt["attempt_index"])

        completed_ids.add(entry["id"])
        progress_summary = summarize_evaluation_results(
            run_name=run_name,
            results=results,
            uses_retry=uses_retry,
            requested_count=len(entries),
        )
        if local_trace_logger is not None:
            progress_summary["local_trace_dir"] = str(local_trace_logger.run_dir)
            local_trace_logger.write_task_result(result)
        write_evaluation_cache(output_path, progress_summary)
        log_bfcl_task_result_to_mlflow(
            result,
            run_name=f"{run_name}_{entry['id']}",
            kind="bfcl_eval_task",
            config=mlflow_config,
            tags=mlflow_tags,
        )

    summary = summarize_evaluation_results(
        run_name=run_name,
        results=results,
        uses_retry=uses_retry,
        requested_count=len(entries),
    )
    if local_trace_logger is not None:
        summary["local_trace_dir"] = str(local_trace_logger.run_dir)
    write_evaluation_cache(output_path, summary)
    if local_trace_logger is not None:
        local_trace_logger.write_summary(summary)
    log_evaluation_summary_to_mlflow(
        summary,
        output_path=output_path,
        kind="bfcl_eval_summary",
        config=mlflow_config,
        tags=mlflow_tags,
    )
    return summary


def teacher_action_policy_messages() -> list[dict[str, str]]:
    return [{"role": "system", "content": TEACHER_ACTION_POLICY_SYSTEM_MESSAGE}]


def teacher_runtime_is_configured(config: TeacherConfig) -> bool:
    if config.provider not in {"mlx_server", "mlx_raw_server"}:
        print(f"Unsupported teacher provider: {config.provider!r}")
        return False

    try:
        with urllib.request.urlopen(f"{config.server_base_url}/health", timeout=10) as response:
            if response.status != 200:
                print(f"MLX teacher server health check returned HTTP {response.status}.")
                return False
            health_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        print(f"Could not reach the MLX teacher server at {config.server_base_url}.")
        print("Start it in another terminal with:")
        if config.provider == "mlx_raw_server":
            print("uv run python scripts/serve_teacher_mlx_raw.py")
        else:
            print("uv run python scripts/serve_teacher_mlx.py")
        print("Error:", error)
        return False
    except json.JSONDecodeError as error:
        print(f"MLX teacher server health check returned invalid JSON: {error}")
        return False

    if config.provider == "mlx_raw_server" and "model" not in health_payload:
        print(f"The server at {config.server_base_url} is alive, but it is not the raw MLX teacher server.")
        print("Start the raw server with:")
        print("uv run python scripts/serve_teacher_mlx_raw.py")
        print("Raw server health should include a model field.")
        return False

    return True


def teacher_server_health(config: TeacherConfig) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{config.server_base_url}/health", timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def list_cached_huggingface_models(prefixes: tuple[str, ...] = ("mlx-community/", "Qwen/")) -> list[str]:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub_dir.exists():
        return []

    model_ids: list[str] = []
    for model_dir in hub_dir.glob("models--*--*"):
        parts = model_dir.name.split("--")
        if len(parts) < 3:
            continue
        model_id = f"{parts[1]}/{'--'.join(parts[2:])}"
        if model_id.startswith(prefixes):
            model_ids.append(model_id)

    return sorted(set(model_ids))


def format_teacher_tool_argument_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def render_qwen_tool_call_from_arguments(function_name: str, arguments: dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={function_name}>"]
    for name, value in arguments.items():
        lines.extend(
            [
                f"<parameter={name}>",
                format_teacher_tool_argument_value(value),
                "</parameter>",
            ]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def mlx_tool_call_to_qwen_text(tool_call: dict[str, Any]) -> str:
    function_payload = tool_call.get("function", {})
    function_name = function_payload.get("name")
    if not function_name:
        raise RuntimeError(f"MLX tool call has no function name: {tool_call}")

    raw_arguments = function_payload.get("arguments", {})
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise RuntimeError(f"MLX tool call arguments are not a dict or JSON string: {tool_call}")

    if not isinstance(arguments, dict):
        raise RuntimeError(f"MLX tool call arguments did not decode to an object: {tool_call}")

    return render_qwen_tool_call_from_arguments(function_name, arguments)


def build_teacher_chat_payload(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    config: TeacherConfig,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "model": config.request_model,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens or config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "stop": GENERATION_STOP_STRINGS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }


def request_mlx_chat_completion(payload: dict[str, Any], config: TeacherConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{config.server_base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MLX teacher HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the MLX teacher server at {config.server_base_url}: {error}") from error


def build_teacher_raw_completion_payload(
    prompt: str,
    config: TeacherConfig,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "model": config.request_model,
        "prompt": prompt,
        "max_tokens": max_tokens or config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "stop": GENERATION_STOP_STRINGS,
    }


def request_mlx_raw_completion(payload: dict[str, Any], config: TeacherConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{config.server_base_url}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception as read_error:
            detail = f"<could not read error body: {read_error}>"
        raise RuntimeError(f"MLX teacher HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the MLX teacher server at {config.server_base_url}: {error}") from error


def record_endpoint_event(
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if endpoint_event_sink is not None:
        endpoint_event_sink(make_json_safe(event))


def qwen_text_from_mlx_raw_completion(response_payload: dict[str, Any]) -> str:
    text = response_payload.get("text")
    if text is None:
        raise RuntimeError(f"MLX raw teacher response has no text field: {response_payload}")
    return str(text)


def qwen_text_from_mlx_chat_response(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"MLX teacher response has no choices: {response_payload}")

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return "\n".join(mlx_tool_call_to_qwen_text(tool_call) for tool_call in tool_calls)

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    if content is not None:
        return str(content)

    if choices[0].get("finish_reason") == "tool_calls":
        return ""

    raise RuntimeError(f"MLX teacher message has neither content, reasoning, nor tool_calls: {choices[0]}")


def mlx_server_chat_completion(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    config: TeacherConfig,
    seed: int | None = None,
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    payload = build_teacher_chat_payload(messages=messages, tools=tools, config=config)
    if seed is not None:
        payload["seed"] = seed
    endpoint_url = f"{config.server_base_url}/v1/chat/completions"
    started_at = time.time()
    try:
        response_payload = request_mlx_chat_completion(payload, config=config)
    except Exception as error:
        record_endpoint_event(
            endpoint_event_sink,
            {
                "provider": config.provider,
                "endpoint_url": endpoint_url,
                "status": "error",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "request_payload": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    record_endpoint_event(
        endpoint_event_sink,
        {
            "provider": config.provider,
            "endpoint_url": endpoint_url,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "request_payload": payload,
            "response_payload": response_payload,
        },
    )
    return qwen_text_from_mlx_chat_response(response_payload)


def mlx_raw_server_completion(
    prompt: str | None,
    config: TeacherConfig,
    seed: int | None = None,
    endpoint_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    if prompt is None:
        raise RuntimeError("The raw MLX teacher provider requires the rendered Qwen prompt.")
    payload = build_teacher_raw_completion_payload(prompt=prompt, config=config)
    if seed is not None:
        payload["seed"] = seed
    endpoint_url = f"{config.server_base_url}/generate"
    started_at = time.time()
    try:
        response_payload = request_mlx_raw_completion(payload, config=config)
    except Exception as error:
        record_endpoint_event(
            endpoint_event_sink,
            {
                "provider": config.provider,
                "endpoint_url": endpoint_url,
                "status": "error",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "request_payload": payload,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    record_endpoint_event(
        endpoint_event_sink,
        {
            "provider": config.provider,
            "endpoint_url": endpoint_url,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started_at, 3),
            "request_payload": payload,
            "response_payload": response_payload,
        },
    )
    return qwen_text_from_mlx_raw_completion(response_payload)


def make_teacher_action_generator(config: TeacherConfig) -> Callable[..., str]:
    def generate_teacher_action(
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        prompt: str | None = None,
    ) -> str:
        generate_teacher_action.last_endpoint_call = None

        def record_call(event: dict[str, Any]) -> None:
            generate_teacher_action.last_endpoint_call = event

        if config.provider == "mlx_raw_server":
            return mlx_raw_server_completion(
                prompt=prompt,
                config=config,
                endpoint_event_sink=record_call,
            )
        if config.provider == "mlx_server":
            return mlx_server_chat_completion(
                messages=messages,
                tools=tools,
                config=config,
                endpoint_event_sink=record_call,
            )
        raise RuntimeError(f"Unsupported teacher provider: {config.provider!r}")

    generate_teacher_action.generation_config = teacher_config_to_dict(config)
    generate_teacher_action.last_endpoint_call = None
    return generate_teacher_action


def make_teacher_action_generator_factory(
    config: TeacherConfig,
) -> Callable[[GenerationAttemptConfig], Callable[..., str]]:
    def make_for_attempt(attempt: GenerationAttemptConfig) -> Callable[..., str]:
        attempt_config = replace(
            config,
            temperature=attempt.temperature,
            top_p=attempt.top_p,
            top_k=attempt.top_k,
        )

        def generate_teacher_action(
            *,
            messages: list[dict[str, str]],
            tools: list[dict[str, Any]],
            prompt: str | None = None,
        ) -> str:
            generate_teacher_action.last_endpoint_call = None

            def record_call(event: dict[str, Any]) -> None:
                generate_teacher_action.last_endpoint_call = event

            if attempt_config.provider == "mlx_raw_server":
                return mlx_raw_server_completion(
                    prompt=prompt,
                    config=attempt_config,
                    seed=attempt.seed,
                    endpoint_event_sink=record_call,
                )
            if attempt_config.provider == "mlx_server":
                return mlx_server_chat_completion(
                    messages=messages,
                    tools=tools,
                    config=attempt_config,
                    seed=attempt.seed,
                    endpoint_event_sink=record_call,
                )
            raise RuntimeError(f"Unsupported teacher provider: {attempt_config.provider!r}")

        generate_teacher_action.generation_config = {
            **teacher_config_to_dict(attempt_config),
            "attempt": attempt_config_to_dict(attempt),
        }
        generate_teacher_action.last_endpoint_call = None
        return generate_teacher_action

    return make_for_attempt


def summarize_trace_for_attempt(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if trace is None:
        return None

    return {
        "decoded_turns": trace.get("decoded_turns", []),
        "trace_rows": [
            {
                "turn": row.get("turn"),
                "step": row.get("step"),
                "stop_reason": row.get("stop_reason"),
                "execution_calls": row.get("execution_calls", []),
                "execution_results": row.get("execution_results", []),
                "parse_errors": row.get("parse_errors", []),
                "raw_output_preview": row.get("raw_output", "")[:500],
            }
            for row in trace.get("trace", [])
        ],
    }


def teacher_trace_to_sft_rows(
    entry: dict[str, Any],
    trace: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for trace_row in trace["trace"]:
        if trace_row.get("stop_reason") != "executed_tool_call":
            continue
        if "prompt" not in trace_row or "assistant_message_content" not in trace_row:
            continue

        assistant_content = trace_row["assistant_message_content"]
        rows.append(
            {
                "split": split_name,
                "id": entry["id"],
                "turn_index": trace_row["turn"],
                "step_index": trace_row["step"],
                "prompt": trace_row["prompt"],
                "completion": assistant_content + "<|im_end|>",
                "assistant_message_content": assistant_content,
                "raw_teacher_output": trace_row["raw_output"],
                "execution_calls": trace_row.get("execution_calls", []),
                "execution_results": trace_row.get("execution_results", []),
                "messages_before_action": trace_row["messages_before_action"],
                "involved_classes": entry["involved_classes"],
            }
        )

    return rows


def write_teacher_collection_cache(
    *,
    attempts_output_path: Path | None,
    traces_output_path: Path | None,
    sft_rows_output_path: Path | None,
    attempt_summaries: list[dict[str, Any]],
    selected_traces: list[dict[str, Any]],
    selected_sft_rows: list[dict[str, Any]],
) -> None:
    if attempts_output_path is not None:
        attempts_output_path.parent.mkdir(parents=True, exist_ok=True)
        attempts_output_path.write_text(
            json.dumps(make_json_safe(attempt_summaries), indent=2),
            encoding="utf-8",
        )
    if traces_output_path is not None:
        traces_output_path.parent.mkdir(parents=True, exist_ok=True)
        traces_output_path.write_text(
            json.dumps(make_json_safe(selected_traces), indent=2),
            encoding="utf-8",
        )
    if sft_rows_output_path is not None:
        write_jsonl(sft_rows_output_path, selected_sft_rows)


def collect_successful_teacher_trajectories(
    entries: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    tokenizer,
    run_limit: int,
    generate_action_factory: Callable[[GenerationAttemptConfig], Callable[..., str]],
    attempt_configs_factory: Callable[[int, dict[str, Any]], list[GenerationAttemptConfig]] = adaptive_temperature_retry_attempts_for_task,
    max_steps_per_turn: int = MAX_STEPS_PER_TURN,
    max_consecutive_execution_errors: int | None = MAX_CONSECUTIVE_EXECUTION_ERRORS,
    attempts_output_path: Path | None = None,
    traces_output_path: Path | None = None,
    sft_rows_output_path: Path | None = None,
    mlflow_config: MlflowConfig | None = None,
    mlflow_tags: dict[str, str] | None = None,
    local_trace_config: LocalTraceConfig | None = None,
) -> dict[str, Any]:
    selected_traces: list[dict[str, Any]] = load_json_file(traces_output_path, []) if traces_output_path else []
    selected_sft_rows: list[dict[str, Any]] = (
        load_jsonl_if_exists(sft_rows_output_path) if sft_rows_output_path else []
    )
    attempt_summaries: list[dict[str, Any]] = (
        load_json_file(attempts_output_path, []) if attempts_output_path else []
    )
    completed_ids = {
        row["id"]
        for row in attempt_summaries
        if isinstance(row, dict) and "id" in row
    }

    if attempt_summaries and attempts_output_path is not None:
        print(f"Loaded {len(attempt_summaries)} cached teacher attempts from: {attempts_output_path}")
    if selected_traces and traces_output_path is not None:
        print(f"Loaded {len(selected_traces)} cached successful teacher traces from: {traces_output_path}")
    if selected_sft_rows and sft_rows_output_path is not None:
        print(f"Loaded {len(selected_sft_rows)} cached teacher SFT rows from: {sft_rows_output_path}")

    local_run_name = (
        attempts_output_path.stem
        if attempts_output_path is not None
        else f"teacher_train_trajectory_collection_{uuid4().hex[:8]}"
    )
    local_trace_logger = create_local_trace_logger(
        local_trace_config,
        run_name=local_run_name,
        kind="teacher_train_trajectory_collection",
        tags=mlflow_tags,
    )
    if local_trace_logger is not None:
        print(f"Local trace directory: {local_trace_logger.run_dir}")
        local_trace_logger.append_event(
            "cache_loaded",
            attempts_output_path=str(attempts_output_path) if attempts_output_path else None,
            traces_output_path=str(traces_output_path) if traces_output_path else None,
            sft_rows_output_path=str(sft_rows_output_path) if sft_rows_output_path else None,
            cached_attempt_count=len(attempt_summaries),
            cached_successful_trace_count=len(selected_traces),
            cached_sft_row_count=len(selected_sft_rows),
        )

    print("Teacher collection max tool-action steps per user turn:", max_steps_per_turn)
    print("Teacher collection max consecutive execution errors:", max_consecutive_execution_errors)

    limited_entries = entries[:run_limit]
    limited_answers = answers[:run_limit]

    for task_index, (entry, answer) in enumerate(zip(limited_entries, limited_answers), start=1):
        assert entry["id"] == answer["id"], (entry["id"], answer["id"])
        if entry["id"] in completed_ids:
            print(f"Teacher task {task_index}/{len(limited_entries)}: {entry['id']} (cached, skipping)")
            if local_trace_logger is not None:
                local_trace_logger.append_event(
                    "task_skipped_cached",
                    task_id=entry["id"],
                    task_index=task_index,
                )
            continue

        print(f"Teacher task {task_index}/{len(limited_entries)}: {entry['id']}")
        if local_trace_logger is not None:
            local_trace_logger.append_event(
                "task_started",
                task_id=entry["id"],
                task_index=task_index,
                involved_classes=entry["involved_classes"],
            )

        retry_result = run_bfcl_entry_harness_with_retry(
            entry,
            answer,
            run_name=f"teacher_train_task_{task_index}",
            tokenizer=tokenizer,
            attempt_configs=attempt_configs_factory(task_index - 1, entry),
            generate_action_factory=generate_action_factory,
            capture_prompts=True,
            prompt_messages_prefix=teacher_action_policy_messages(),
            max_steps_per_turn=max_steps_per_turn,
            max_consecutive_execution_errors=max_consecutive_execution_errors,
            local_trace_logger=local_trace_logger,
        )

        for attempt_summary in retry_result["attempts"]:
            summary = {
                "id": entry["id"],
                "task_index": task_index,
                **attempt_summary,
            }
            attempt_summaries.append(summary)
            print(
                f"  attempt {attempt_summary['attempt_index']}: "
                f"{summarize_score(attempt_summary['score'])} "
                f"({attempt_summary['elapsed_seconds']}s)"
            )
            if attempt_summary.get("error_message"):
                print("    error:", attempt_summary["error_message"][:500])

        if retry_result["valid"]:
            selected_attempt = retry_result["selected_attempt"] or {}
            selected_traces.append(
                {
                    "id": entry["id"],
                    "attempt_index": selected_attempt.get("attempt_index"),
                    "attempt_config": selected_attempt.get("attempt_config"),
                    "score": make_json_safe(retry_result["score"]),
                    "trace": make_json_safe(retry_result["trace"]),
                }
            )
            selected_sft_rows.extend(
                teacher_trace_to_sft_rows(entry, retry_result["trace"], split_name="teacher_train")
            )
        else:
            print("  no successful teacher trajectory kept for this task")

        completed_ids.add(entry["id"])
        local_task_result = {
            "id": entry["id"],
            "valid": retry_result["valid"],
            "score": make_json_safe(retry_result["score"]),
            "trace": make_json_safe(retry_result["trace"]),
            "attempts": make_json_safe(retry_result["attempts"]),
            "attempt_traces": make_json_safe(retry_result.get("attempt_traces", [])),
            "selected_attempt": make_json_safe(retry_result.get("selected_attempt")),
        }
        if local_trace_logger is not None:
            local_trace_logger.write_task_result(local_task_result)
        log_bfcl_task_result_to_mlflow(
            local_task_result,
            run_name=f"teacher_train_{entry['id']}",
            kind="teacher_train_trajectory_task",
            config=mlflow_config,
            tags=mlflow_tags,
        )
        write_teacher_collection_cache(
            attempts_output_path=attempts_output_path,
            traces_output_path=traces_output_path,
            sft_rows_output_path=sft_rows_output_path,
            attempt_summaries=attempt_summaries,
            selected_traces=selected_traces,
            selected_sft_rows=selected_sft_rows,
        )

    write_teacher_collection_cache(
        attempts_output_path=attempts_output_path,
        traces_output_path=traces_output_path,
        sft_rows_output_path=sft_rows_output_path,
        attempt_summaries=attempt_summaries,
        selected_traces=selected_traces,
        selected_sft_rows=selected_sft_rows,
    )

    summary = {
        "attempts": attempt_summaries,
        "selected_traces": selected_traces,
        "sft_rows": selected_sft_rows,
    }
    if local_trace_logger is not None:
        summary["local_trace_dir"] = str(local_trace_logger.run_dir)
        local_trace_logger.write_summary(summary)
    log_teacher_collection_to_mlflow(
        summary,
        attempts_output_path=attempts_output_path,
        traces_output_path=traces_output_path,
        sft_rows_output_path=sft_rows_output_path,
        config=mlflow_config,
        tags=mlflow_tags,
    )
    return summary
