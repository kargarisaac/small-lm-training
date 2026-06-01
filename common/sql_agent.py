from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download

from . import config as cfg


DATASET = cfg.SQL_AGENT_DATASET
SYSTEM_PROMPT = """You are a SQL tool-use agent.
You repair or write SQLite for a user issue by interacting with a deterministic database environment.

Return exactly one structured JSON decision per assistant turn. Do not write prose outside JSON.

Decision shape:
{"draft": "short plan, max 18 words", "output": {"action": "inspect_schema"}}
{"draft": "short plan, max 18 words", "output": {"action": "run_sql_query", "sql": "SELECT ..."}}
{"draft": "short plan, max 18 words", "output": {"action": "submit_sql", "sql": ["SQL statement 1", "SQL statement 2"]}}

Available actions:
{"action": "inspect_schema"}
{"action": "run_sql_query", "sql": "SELECT ..."}
{"action": "submit_sql", "sql": ["SQL statement 1", "SQL statement 2"]}

Rules:
- Use inspect_schema before writing SQL if the schema is not obvious.
- Use run_sql_query to test SQL and inspect errors or result rows.
- Use submit_sql only when you are ready to submit the final corrected SQL.
- The submitted SQL must be SQLite.
- If the final answer needs multiple statements, submit a JSON list in order.
"""


Generate = Callable[[list[dict[str, str]]], str]


@dataclass
class EvalSummary:
    total: int
    success: int
    parsed_actions: int
    submitted: int
    sql_execution_errors: int
    max_turn_failures: int
    parse_failures: int
    repeated_action_failures: int
    runtime_errors: int
    average_turns: float

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0


def normalize_source_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["instance_id"],
        "instance_id": row["instance_id"],
        "db_id": row["db_id"],
        "category": row["category"],
        "query": row["query"],
        "issue_sql": list(row["issue_sql"] or []),
        "sol_sql": list(row["sol_sql"] or []),
        "preprocess_sql": list(row["preprocess_sql"] or []),
        "clean_up_sql": list(row["clean_up_sql"] or []),
        "test_cases": list(row["test_cases"] or []),
    }


def category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def copy_database_templates(output_dir: Path, db_ids: set[str]) -> None:
    for db_id in sorted(db_ids):
        source = hf_hub_download(DATASET, f"database/{db_id}/{db_id}_template.sqlite", repo_type="dataset")
        dest = template_path(output_dir, db_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def template_path(data_dir: Path, db_id: str) -> Path:
    return data_dir / "dbs" / db_id / f"{db_id}_template.sqlite"


def load_rows(data_dir: Path, partition: str, limit: int | None = None) -> list[dict[str, Any]]:
    path = data_dir / f"{partition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Prepared SQL-agent file not found: {path}. Run notebook 01 to explore and write the split first.")
    return cfg.read_jsonl(path, limit)


def initial_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    issue_sql = "\n\n".join(row["issue_sql"]) if row["issue_sql"] else "(none provided)"
    content = f"""Database id: {row["db_id"]}
Task category: {row["category"]}

User issue:
{row["query"]}

Buggy or incomplete SQL from the user:
```sql
{issue_sql}
```

Use the database tools to inspect, test, and submit corrected SQL."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parse_decision(text: str) -> tuple[str | None, dict[str, Any] | None]:
    stripped = strip_model_text(text)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = first_json_object(stripped)
        if value is None:
            return None, None
    if not isinstance(value, dict):
        return None, None
    draft = value.get("draft") if isinstance(value.get("draft"), str) else None
    if isinstance(value.get("output"), dict):
        value = value["output"]
    action = value.get("action")
    if action not in {"inspect_schema", "run_sql_query", "submit_sql"}:
        return draft, None
    normalized: dict[str, Any] = {"action": action}
    if action == "run_sql_query" and not isinstance(value.get("sql"), str):
        return draft, None
    if action == "run_sql_query":
        normalized["sql"] = value["sql"]
    if action == "submit_sql":
        sql = value.get("sql")
        if isinstance(sql, str):
            normalized["sql"] = [sql]
        elif not (isinstance(sql, list) and all(isinstance(item, str) for item in sql)):
            return draft, None
        else:
            normalized["sql"] = sql
    return draft, normalized


def first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def strip_model_text(text: str) -> str:
    stripped = text.strip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        stripped = stripped.replace(token, "").strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return stripped


def run_task(
    row: dict[str, Any],
    *,
    data_dir: Path,
    generate: Generate,
    max_turns: int = 8,
    keep_messages: bool = False,
) -> dict[str, Any]:
    messages = initial_messages(row)
    trace = []
    schema_observation: str | None = None
    previous_action_key: str | None = None
    with task_database(row, data_dir) as db_path:
        conn = sqlite3.connect(db_path)
        try:
            execute_sql_list(conn, row["preprocess_sql"])
            for turn in range(1, max_turns + 1):
                messages_before = copy.deepcopy(messages)
                baml_output = generate(messages_before)
                draft, action = parse_decision(baml_output)
                trace_item: dict[str, Any] = {"turn": turn, "baml_output": baml_output, "draft": draft, "action": action}
                if keep_messages:
                    trace_item["messages_before"] = messages_before
                messages.append({"role": "assistant", "content": baml_output})
                if action is None:
                    trace_item["stop_reason"] = "parse_failure"
                    trace.append(trace_item)
                    return task_result(row, trace, "parse_failure", False)
                action_key = json.dumps(action, sort_keys=True, ensure_ascii=False)
                if action_key == previous_action_key:
                    trace_item["stop_reason"] = "repeated_action"
                    trace.append(trace_item)
                    return task_result(row, trace, "repeated_action", False)
                previous_action_key = action_key
                if action["action"] == "inspect_schema":
                    if schema_observation is None:
                        schema_observation = schema_text(conn)
                        observation = schema_observation
                    else:
                        observation = "Schema was already provided earlier in this task. Use the previous schema observation instead of calling inspect_schema again."
                    trace_item["observation"] = observation
                    messages.append({"role": "user", "content": environment_message(observation)})
                    trace_item["stop_reason"] = "tool_observation"
                    trace.append(trace_item)
                    continue
                if action["action"] == "run_sql_query":
                    observation = run_sql_observation(conn, action["sql"])
                    trace_item["observation"] = observation
                    messages.append({"role": "user", "content": environment_message(json.dumps(observation, ensure_ascii=False))})
                    trace_item["stop_reason"] = "tool_observation"
                    trace.append(trace_item)
                    continue
                score = evaluate_submitted_sql(row, action["sql"], data_dir)
                trace_item["score"] = score
                trace_item["stop_reason"] = "submitted"
                trace.append(trace_item)
                return task_result(row, trace, "submitted", score["success"])
            return task_result(row, trace, "max_turns", False)
        finally:
            conn.close()


def task_result(row: dict[str, Any], trace: list[dict[str, Any]], stop_reason: str, success: bool) -> dict[str, Any]:
    return {
        "id": row["id"],
        "db_id": row["db_id"],
        "category": row["category"],
        "success": success,
        "stop_reason": stop_reason,
        "turns": len(trace),
        "trace": trace,
    }


def environment_message(observation: str) -> str:
    return f"Environment observation:\n{observation}\n\nReturn the next structured JSON decision."


@contextlib.contextmanager
def task_database(row: dict[str, Any], data_dir: Path):
    source = template_path(data_dir, row["db_id"])
    if not source.exists():
        raise FileNotFoundError(f"Missing database template: {source}")
    with tempfile.TemporaryDirectory(prefix="sql_agent_") as tmp:
        target = Path(tmp) / f"{row['db_id']}.sqlite"
        shutil.copy2(source, target)
        yield target


def schema_text(conn: sqlite3.Connection) -> str:
    rows = [
        ("main", name, create_sql)
        for name, create_sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    ]
    rows.extend(
        ("temp", name, create_sql)
        for name, create_sql in conn.execute("SELECT name, sql FROM sqlite_temp_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    )
    chunks = []
    for schema_name, table, create_sql in rows:
        chunks.append(f"-- {schema_name} table\n{create_sql or f'TABLE {table}'}")
        columns = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
        col_text = ", ".join(f"{col[1]} {col[2]}".strip() for col in columns)
        chunks.append(f"columns: {col_text}")
    return "\n\n".join(chunks)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def run_sql_observation(conn: sqlite3.Connection, sql: str) -> dict[str, Any]:
    try:
        result = execute_one_sql(conn, sql)
        rows = result if isinstance(result, list) else []
        return {"ok": True, "rows": cfg.make_json_safe(rows[:20]), "row_count": len(rows), "truncated": len(rows) > 20}
    except Exception as error:
        conn.rollback()
        return {"ok": False, "error": str(error)}


def execute_one_sql(conn: sqlite3.Connection, sql: str) -> list[Any] | None:
    cur = conn.execute(sql)
    if sql.lstrip().lower().startswith(("select", "with", "pragma")):
        return cur.fetchall()
    conn.commit()
    try:
        return cur.fetchall()
    except sqlite3.Error:
        return None


def execute_sql_list(conn: sqlite3.Connection, sqls: list[str]) -> tuple[list[Any] | None, bool, str]:
    last = None
    for sql in sqls:
        try:
            last = execute_one_sql(conn, sql)
        except Exception as error:
            conn.rollback()
            return last, True, str(error)
    return last, False, ""


def evaluate_submitted_sql(row: dict[str, Any], pred_sqls: list[str], data_dir: Path) -> dict[str, Any]:
    with task_database(row, data_dir) as db_path:
        conn = sqlite3.connect(db_path)
        try:
            _, preprocess_error, preprocess_message = execute_sql_list(conn, row["preprocess_sql"])
            if preprocess_error:
                return {"success": False, "error": f"preprocess_error: {preprocess_message}", "pred_sqls": pred_sqls}
            pred_result, execution_error, execution_message = execute_sql_list(conn, pred_sqls)
            if execution_error:
                return {"success": False, "error": f"execution_error: {execution_message}", "pred_sqls": pred_sqls}
            test_cases = row["test_cases"] or [default_test_case()]
            failures = []
            for index, test_code in enumerate(test_cases, start=1):
                passed, message = run_test_case(test_code, pred_result, pred_sqls, row["sol_sql"], db_path, conn)
                if not passed:
                    failures.append({"test": index, "message": message})
            return {"success": not failures, "failures": failures, "pred_sqls": pred_sqls}
        finally:
            conn.close()


def run_test_case(
    test_code: str,
    pred_query_result: Any,
    pred_sqls: list[str],
    sol_sqls: list[str],
    db_path: Path,
    conn: sqlite3.Connection,
) -> tuple[bool, str]:
    global_env = {
        "sqlite3": sqlite3,
        "json": json,
        "date": date,
        "datetime": datetime,
        "execute_queries": lambda sqls, path, connection=None, logger=None, section_title="", return_error=False: test_execute_queries(sqls, path, connection or conn, return_error),
        "perform_query_on_sqlite_databases": lambda query, path, conn=None, query_timeout=30: (execute_one_sql(conn or sqlite3.connect(path), query), conn),
        "ex_base": ex_base,
        "check_sql_function_usage": check_sql_function_usage,
        "remove_distinct": remove_distinct,
        "remove_comments": remove_comments,
        "remove_round": remove_round,
        "preprocess_results": preprocess_results,
        "pred_query_result": pred_query_result,
    }
    local_env = {
        "pred_sqls": pred_sqls,
        "sol_sqls": sol_sqls,
        "db_path": str(db_path),
        "conn": conn,
        "conditions": {},
    }
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec("import datetime\nfrom datetime import date\n" + test_code + "\n__result__ = test_case(pred_sqls, sol_sqls, db_path, conn, conditions)", global_env, local_env)
        return True, ""
    except AssertionError as error:
        return False, str(error)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def test_execute_queries(sqls: list[str] | str, db_path: str | Path, conn: sqlite3.Connection, return_error: bool = False):
    result, has_error, message = execute_sql_list(conn, [sqls] if isinstance(sqls, str) else list(sqls))
    if return_error:
        return result, has_error, False, message
    return result, has_error, False


def default_test_case() -> str:
    return """
def test_case(pred_sqls, sol_sqls, db_path, conn, conditions):
    result = ex_base(pred_sqls, sol_sqls, db_path, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result
"""


def ex_base(pred_sqls: list[str], sol_sqls: list[str], db_path: str | Path, conn: sqlite3.Connection, conditions: dict[str, Any] | None = None) -> int:
    if not pred_sqls or not sol_sqls:
        return 0
    pred_result, pred_error, _ = execute_sql_list(conn, pred_sqls)
    sol_result, sol_error, _ = execute_sql_list(conn, sol_sqls)
    if pred_error or sol_error or pred_result is None or sol_result is None:
        return 0
    pred_processed = preprocess_results(pred_result)
    sol_processed = preprocess_results(sol_result)
    if conditions and conditions.get("order"):
        return int(pred_processed == sol_processed)
    return int(set(pred_processed) == set(sol_processed))


def preprocess_results(results: list[Any], decimal_places: int = 2) -> list[tuple[Any, ...]]:
    processed = []
    for row in results:
        values = row if isinstance(row, (tuple, list)) else (row,)
        processed.append(tuple(normalize_value(value, decimal_places) for value in values))
    return processed


def normalize_value(value: Any, decimal_places: int) -> Any:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Decimal):
        return value.quantize(Decimal(1).scaleb(-decimal_places), rounding=ROUND_HALF_UP)
    if isinstance(value, float):
        return round(value, decimal_places)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bytes):
        return value.hex()
    return value


def remove_distinct(sqls: list[str]) -> list[str]:
    return [re.sub(r"\bDISTINCT\b", "", sql, flags=re.IGNORECASE) for sql in sqls]


def remove_comments(sqls: list[str]) -> list[str]:
    cleaned = []
    for sql in sqls:
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        sql = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", sql)
        cleaned.append(sql.strip())
    return cleaned


def remove_round(sqls: list[str]) -> list[str]:
    return [re.sub(r"ROUND\s*\(([^,()]+)(?:,[^)]+)?\)", r"\1", sql, flags=re.IGNORECASE) for sql in sqls]


def check_sql_function_usage(sqls: list[str], required_keywords: list[str]) -> int:
    combined = " ".join(sql.lower() for sql in sqls)
    return int(all(keyword.lower() in combined for keyword in required_keywords))


def summarize_results(results: list[dict[str, Any]]) -> EvalSummary:
    total = len(results)
    return EvalSummary(
        total=total,
        success=sum(bool(row["success"]) for row in results),
        parsed_actions=sum(sum(1 for item in row["trace"] if item.get("action") is not None) for row in results),
        submitted=sum(row["stop_reason"] == "submitted" for row in results),
        sql_execution_errors=sum(any((item.get("score") or {}).get("error", "").startswith("execution_error") for item in row["trace"]) for row in results),
        max_turn_failures=sum(row["stop_reason"] == "max_turns" for row in results),
        parse_failures=sum(row["stop_reason"] == "parse_failure" for row in results),
        repeated_action_failures=sum(row["stop_reason"] == "repeated_action" for row in results),
        runtime_errors=sum(row["stop_reason"].endswith("runtime_error") or row["stop_reason"] == "runtime_error" for row in results),
        average_turns=sum(row["turns"] for row in results) / total if total else 0.0,
    )


def successful_sft_trace_rows(task_result: dict[str, Any], teacher_model: str, teacher_backend: str) -> list[dict[str, Any]]:
    if not task_result["success"]:
        return []
    rows = []
    for index, item in enumerate(task_result["trace"], start=1):
        if "messages_before" not in item or item.get("action") is None:
            continue
        rows.append(
            {
                "id": f"{task_result['id']}_turn_{index}",
                "task_id": task_result["id"],
                "db_id": task_result["db_id"],
                "category": task_result["category"],
                "messages": item["messages_before"] + [{"role": "assistant", "content": item["baml_output"]}],
                "teacher_baml_output": item["baml_output"],
                "teacher_draft": item.get("draft"),
                "teacher_action": item["action"],
                "teacher_model": teacher_model,
                "teacher_backend": teacher_backend,
            }
        )
    return rows


def split_train_validation(rows: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    validation_size = max(1, int(len(rows) * fraction))
    return rows[validation_size:], rows[:validation_size]


def write_mlx_lm_data(data_dir: Path, train_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_jsonl(data_dir / "train.jsonl", [{"messages": row["messages"]} for row in train_rows])
    cfg.write_jsonl(data_dir / "valid.jsonl", [{"messages": row["messages"]} for row in valid_rows])
    cfg.write_jsonl(data_dir / "test.jsonl", [{"messages": row["messages"]} for row in valid_rows])


def canonical_decision_text(draft: str | None, action: dict[str, Any]) -> str:
    return json.dumps(
        {"draft": draft or "Choose the next executable SQL-agent action.", "output": action},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def render_baml_sql_agent_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    from baml_py import ClientRegistry
    from baml_client import b
    from baml_client.types import SqlAgentMessage

    client_registry = ClientRegistry()
    client_registry.add_llm_client(
        "SqlAgentRuntimeClient",
        "openai-generic",
        {"base_url": "http://127.0.0.1:1/v1", "model": "sql-agent-sft-render", "api_key": "dummy"},
    )
    client_registry.set_primary("SqlAgentRuntimeClient")
    request = b.with_options(client_registry=client_registry).request.SqlAgentNextAction(
        [SqlAgentMessage(role=message["role"], content=message["content"]) for message in messages],
    )
    body = request.body.json()
    rendered = body.get("messages")
    if not isinstance(rendered, list) or not all(isinstance(message, dict) for message in rendered):
        raise RuntimeError(f"BAML rendered an unexpected SQL-agent request body: {body!r}")
    return [{"role": str(message["role"]), "content": str(message["content"])} for message in rendered]


def canonical_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("sft_prompt_format") == "baml_sql_agent":
        return row
    if "teacher_action" not in row:
        raise ValueError(f"SFT row {row.get('id', '<unknown>')} is missing teacher_action.")
    messages = list(row["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"SFT row {row.get('id', '<unknown>')} must end with an assistant target message.")
    target = canonical_decision_text(row.get("teacher_draft"), row["teacher_action"])
    canonical_messages = []
    for message in messages[:-1]:
        if message.get("role") != "assistant":
            canonical_messages.append(message)
            continue
        draft, action = parse_decision(message.get("content", ""))
        if action is None:
            raise ValueError(f"SFT row {row.get('id', '<unknown>')} contains an unparseable previous assistant action.")
        canonical_messages.append({"role": "assistant", "content": canonical_decision_text(draft, action)})
    return row | {
        "messages": render_baml_sql_agent_messages(canonical_messages) + [{"role": "assistant", "content": target}],
        "sft_prompt_format": "baml_sql_agent",
        "sft_target_format": "baml_decision_json",
    }


def prepare_sft_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    validation_fraction: float,
) -> dict[str, Any]:
    lengths = []
    kept_rows = []
    for row in rows:
        canonical = canonical_sft_row(row)
        text = tokenizer.apply_chat_template(
            canonical["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=cfg.QWEN_ENABLE_THINKING,
        )
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(length)
        if length <= max_length:
            kept_rows.append(canonical)
    train_rows, valid_rows = split_train_validation(kept_rows, validation_fraction)
    sorted_lengths = sorted(lengths)
    stats = {
        "source_rows": len(rows),
        "kept_rows": len(kept_rows),
        "dropped_rows": len(rows) - len(kept_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "max_seq_length": max_length,
    }
    if sorted_lengths:
        for name, fraction in {"p50": 0.50, "p90": 0.90, "p95": 0.95}.items():
            stats[name] = sorted_lengths[round((len(sorted_lengths) - 1) * fraction)]
        stats["min"] = sorted_lengths[0]
        stats["max"] = sorted_lengths[-1]
    return {"rows": kept_rows, "train_rows": train_rows, "valid_rows": valid_rows, "stats": stats}


def tokenize_sft_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any] | None:
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg.QWEN_ENABLE_THINKING,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=cfg.QWEN_ENABLE_THINKING,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
