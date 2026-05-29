from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import ast


TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
SPECIAL_TOKENS = [
    "<|startoftext|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
]


@dataclass
class ParsedLiquidToolCall:
    name: str
    arguments: dict[str, Any]
    raw_call: str


@dataclass
class LiquidToolParseResult:
    calls: list[ParsedLiquidToolCall]
    errors: list[str]


def _tool_call_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    cursor = 0
    while True:
        start = text.find(TOOL_CALL_START, cursor)
        if start < 0:
            return blocks
        block_start = start + len(TOOL_CALL_START)
        end = text.find(TOOL_CALL_END, block_start)
        if end < 0:
            blocks.append(text[block_start:].strip())
            return blocks
        blocks.append(text[block_start:end].strip())
        cursor = end + len(TOOL_CALL_END)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    raise ValueError(f"Unsupported function node: {ast.dump(node)}")


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id == "null":
            return None
    return ast.literal_eval(node)


def _parse_call_node(node: ast.AST, raw_call: str) -> ParsedLiquidToolCall:
    if not isinstance(node, ast.Call):
        raise ValueError(f"Expected function call, got: {ast.dump(node)}")
    if node.args:
        raise ValueError("Liquid tool calls must use named keyword arguments.")

    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError("Liquid tool calls do not support **kwargs.")
        arguments[keyword.arg] = _literal_value(keyword.value)

    return ParsedLiquidToolCall(
        name=_call_name(node.func),
        arguments=arguments,
        raw_call=raw_call,
    )


def parse_liquid_tool_calls(text: str) -> LiquidToolParseResult:
    calls: list[ParsedLiquidToolCall] = []
    errors: list[str] = []

    for block in _tool_call_blocks(text):
        if not block:
            continue
        try:
            expression = ast.parse(block, mode="eval").body
            nodes = expression.elts if isinstance(expression, (ast.List, ast.Tuple)) else [expression]
            for node in nodes:
                raw_call = ast.unparse(node) if hasattr(ast, "unparse") else block
                calls.append(_parse_call_node(node, raw_call))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}; block={block!r}")

    if not calls and not errors and TOOL_CALL_START in text:
        errors.append("Tool-call marker found, but no parseable Liquid tool call was produced.")

    return LiquidToolParseResult(calls=calls, errors=errors)


def strip_liquid_generated_special_tokens(text: str) -> str:
    cleaned = text
    for token in SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")

    while "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>", start)
        if end < 0:
            break
        cleaned = cleaned[:start] + cleaned[end + len("</think>") :]

    return cleaned.strip()
