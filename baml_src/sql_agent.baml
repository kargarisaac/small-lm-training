class SqlAgentMessage {
  role "system" | "user" | "assistant"
  content string
}

class InspectSchemaAction {
  action "inspect_schema"
}

class RunSqlQueryAction {
  action "run_sql_query"
  sql string @description("SQLite query to execute against the temporary task database.")
}

class SubmitSqlAction {
  action "submit_sql"
  sql string[] @description("Final SQLite statement or statements to submit for hidden tests.")
}

class SqlAgentDecision {
  draft string @description("Short chain-of-draft plan for this single next step. Keep it under 18 words.")
  output InspectSchemaAction | RunSqlQueryAction | SubmitSqlAction @description("Exactly one executable harness action.")
}

function SqlAgentNextAction(messages: SqlAgentMessage[]) -> SqlAgentDecision {
  client MlxCommunityQwen35_08bMlxBf16
  prompt #"
    {{ _.role("system") }}
    Decide the next SQL-agent step.
    Return a short chain-of-draft in `draft`, then exactly one executable action in `output`.
    Do not write prose outside the structured output.
    Do not repeat an observation query once the environment has already returned the needed rows.
    Use submit_sql when the current context is enough to submit a final answer.

    {{ ctx.output_format }}

    Conversation:

    {% for message in messages %}
      {{ _.role(message.role) }}
      {{ message.content }}
    {% endfor %}
  "#
}

test InspectSchemaDecision {
  functions [SqlAgentNextAction]
  args {
    messages [
      {
        role "system"
        content "You are a SQL tool-use agent. Return a structured decision."
      }
      {
        role "user"
        content "Database id: demo\n\nUser issue:\nFix this query.\n\nBuggy SQL:\nselect * from orders;\n\nNo schema has been observed yet."
      }
    ]
  }
}

test SubmitSqlDecision {
  functions [SqlAgentNextAction]
  args {
    messages [
      {
        role "system"
        content "You are a SQL tool-use agent. Return a structured decision."
      }
      {
        role "user"
        content "Database id: demo\n\nUser issue:\nCount rows in orders.\n\nBuggy SQL:\nselect count(*) from order;"
      }
      {
        role "assistant"
        content "{\"draft\":\"Need schema first.\",\"output\":{\"action\":\"inspect_schema\"}}"
      }
      {
        role "user"
        content "Environment observation:\nCREATE TABLE orders (id integer primary key, amount real);\n\nReturn the next structured JSON decision."
      }
      {
        role "assistant"
        content "{\"draft\":\"Use the observed table name.\",\"output\":{\"action\":\"run_sql_query\",\"sql\":\"select count(*) from orders;\"}}"
      }
      {
        role "user"
        content "Environment observation:\n{\"ok\": true, \"rows\": [[3]], \"row_count\": 1, \"truncated\": false}\n\nReturn the next structured JSON decision."
      }
    ]
  }
}
