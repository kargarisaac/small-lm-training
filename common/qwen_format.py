from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json


__all__ = [
    "ParsedQwenToolCall",
    "QwenParseResult",
    "parse_qwen_tool_calls",
    "strip_generated_special_tokens",
    "qwen_text_from_tool_call_parts",
]


@dataclass
class ParsedQwenToolCall:
    name: str
    arguments: dict[str, str]
    raw_block: str


@dataclass
class QwenParseResult:
    calls: list[ParsedQwenToolCall]
    errors: list[str]


def _extract_tag_blocks(text: str, start_tag: str, end_tag: str) -> tuple[list[str], list[str]]:
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


def _parse_qwen_opening_tag(line: str, prefix: str, label: str) -> tuple[str | None, str, str | None]:
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


def _split_inline_parameter_close(value: str) -> tuple[str, bool, str]:
    close_tag = "</parameter>"
    if close_tag not in value:
        return value, False, ""

    parameter_value, trailing = value.split(close_tag, 1)
    return parameter_value, True, trailing.strip()


def _parse_qwen_tool_call_block(block: str) -> tuple[ParsedQwenToolCall | None, list[str]]:
    normalized_block = (
        block.replace("><function=", ">\n<function=")
        .replace("> <function=", ">\n<function=")
        .replace("></function>", ">\n</function>")
        .replace("> </function>", ">\n</function>")
        .replace("><parameter=", ">\n<parameter=")
        .replace("> <parameter=", ">\n<parameter=")
        .replace("></parameter>", ">\n</parameter>")
        .replace("> </parameter>", ">\n</parameter>")
    )
    lines = [line.strip() for line in normalized_block.splitlines() if line.strip()]
    errors: list[str] = []

    if not lines:
        return None, ["Empty tool_call block."]

    function_line = lines[0]
    function_name, function_inline_value, function_error = _parse_qwen_opening_tag(
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

        parameter_name, inline_value, parameter_error = _parse_qwen_opening_tag(
            line,
            "<parameter=",
            "parameter",
        )
        if parameter_error is not None:
            errors.append(parameter_error)
            index += 1
            continue

        value_lines: list[str] = []
        inline_value, parameter_closed_inline, trailing_after_close = _split_inline_parameter_close(inline_value)
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
    blocks, errors = _extract_tag_blocks(text, "<tool_call>", "</tool_call>")
    calls: list[ParsedQwenToolCall] = []

    for block in blocks:
        call, block_errors = _parse_qwen_tool_call_block(block)
        errors.extend(block_errors)
        if call is not None:
            calls.append(call)

    if not blocks:
        errors.append("No <tool_call> block found.")

    return QwenParseResult(calls=calls, errors=errors)


def strip_generated_special_tokens(text: str) -> str:
    cleaned = text.split("<|im_end|>", 1)[0].strip()
    for special_token in ["<|endoftext|>"]:
        cleaned = cleaned.replace(special_token, "")
    return cleaned.strip()


def qwen_text_from_tool_call_parts(function_name: str, arguments: dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={function_name}>"]
    for parameter_name, value in arguments.items():
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = str(value)
        lines.extend([f"<parameter={parameter_name}>", value_text, "</parameter>"])
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)
