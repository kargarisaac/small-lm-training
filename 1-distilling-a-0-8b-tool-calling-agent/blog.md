# Distilling A 0.8B SQL Tool-Use Agent

I started this experiment with a very specific question:

> Can a tiny model learn to act like a stronger model inside a real tool-use loop?

Not "can it write SQL-looking text?" Not "can it pass a prompt demo?" I wanted the small model to sit inside the same harness as the teacher, inspect a database, run SQL, read observations, and submit a final answer that passes deterministic hidden tests.

That difference matters. Distilling a normal chat answer is one thing. Distilling an agent is messier, because the thing we want to transfer is not only a final string. It is a sequence of decisions.

This post is the first version of that story. I kept the benchmark fixed, kept the environment fixed, and changed only the model/training side. The runnable commands live in the README. Here I want to explain the theory, the architecture, the harness, and what actually happened in the experiments.

![Experiment story](assets/experiment-story-sketchnote.png)

## Distillation, Plainly

Knowledge distillation is the idea that a stronger teacher model can transfer some useful behavior into a smaller student model.

The classic version is simple: train the student to match the teacher. But there are several different things the student can match:

| Distillation signal | What the student sees | Typical use |
| --- | --- | --- |
| Hard labels | The teacher's chosen answer tokens | Supervised fine-tuning, instruction tuning |
| Soft labels / logits | The teacher's probability distribution over tokens | More information about alternatives and uncertainty |
| Hidden states | Internal activations from a teacher network | Usually closer architecture families or special training setups |
| Trajectories | Multi-step behavior: actions, observations, final answer | Agents and tool-use systems |
| Rewards | A score for the result, sometimes with rollouts | RL-style training after or instead of SFT |

This blog uses **offline hard-token trajectory distillation**.

"Offline" means I first ran the teacher and saved the traces. The student never asks the teacher for help during training. "Hard-token" means the target is the teacher's actual next action, not the teacher's full probability distribution. "Trajectory" means a successful task can produce multiple training examples: one for inspecting the schema, one for running SQL, one for submitting the final answer, and so on.

![Distillation signals](assets/distillation-signals-sketchnote.png)

For an ordinary chat task, a training row might look like:

```text
prompt -> answer
```

For this agent task, a training row looks more like:

```text
conversation so far -> next structured action
```

If the teacher solves one SQL task in three turns, that single task can become three SFT rows:

```text
user issue -> inspect schema
user issue + schema observation -> run SQL
user issue + schema + SQL result -> submit final SQL
```

That is the core of the experiment.

## The Fixed World

The benchmark is [`birdsql/six-gym-sqlite`](https://huggingface.co/datasets/birdsql/six-gym-sqlite). Each task contains a user issue, a buggy or incomplete SQL query, a SQLite database, optional setup SQL, hidden tests, and reference SQL.

The model does not see the hidden tests or the reference SQL. It only sees the user issue and the tool observations it earns by acting in the environment.

![One SQL-agent task](assets/sql-agent-task-sketchnote.png)

For this post I narrowed the task distribution instead of trying to cover everything at once:

| Setting | Value |
| --- | --- |
| Task category | `Query` |
| Databases | `netflix`, `movie_3`, `books`, `chinook` |
| Source rows scanned | 5000 |
| Candidate rows after filtering | 1099 |
| Train split | 879 tasks |
| Eval split | 220 tasks |
| Split seed | 42 |

This is intentionally a small world. That makes it easier to ask a clean question: within one fixed SQL repair environment, how much agent behavior can we transfer into a small model?

## The Harness

The harness gives the model exactly three actions:

```json
{"action": "inspect_schema"}
{"action": "run_sql_query", "sql": "SELECT ..."}
{"action": "submit_sql", "sql": ["SQL statement 1", "SQL statement 2"]}
```

The loop is deterministic. The model emits one structured JSON action. The harness executes it. If the action was `inspect_schema` or `run_sql_query`, the harness appends an environment observation and asks for the next action. If the action was `submit_sql`, the harness runs the hidden tests and stops.

![Fixed SQL harness](assets/sql-agent-fixed-harness-sketchnote.png)

There is no LLM judge here. The score is not a preference model saying whether the answer looks good. The score is whether the submitted SQL passes the hidden tests against the SQLite database.

That gives the eval a useful shape:

```text
user issue
  -> model action
  -> SQLite observation
  -> model action
  -> hidden tests
  -> pass/fail
```

The structured-output layer matters because tool-use failures are often boring but fatal. A model can be smart enough to reason about the SQL and still fail the task if it prints prose, repeats the same action, or never submits. So the harness records not just success, but also how the run stopped: submitted, parse failure, repeated action, max turns, or runtime error.

I did not change that harness for the comparisons below.

## The Model Architecture

The runtime architecture has two sides.

The **teacher side** runs stronger models inside the same SQL harness. If a teacher solves a train task, I keep the trace. If it fails, I do not use that trace for SFT.

The **student side** is a smaller chat model trained with LoRA on those successful teacher actions. The student is still just a decoder-only language model predicting next tokens, but the target text is the next executable JSON action, not a friendly natural-language answer.

There are also two hardware paths:

| Path | Role |
| --- | --- |
| Mac + MLX | Local serving and eval for models that fit |
| Rented NVIDIA GPU | Student SFT runs with Unsloth-style training |

The teacher/student setup I used in this round:

| Role | Models |
| --- | --- |
| Strong teachers / teacher candidates | GPT 5.5 medium, Qwen3.5-35B-A3B 8-bit |
| Main tiny student | Qwen3.5-0.8B |
| Nearby student comparisons | Qwen3.5-2B, LFM2.5-8B-A1B |

The point of the nearby comparisons was not to crown a universal best model. It was to see whether the result was specific to the 0.8B student, whether a 2B same-family student changed the picture, and whether a larger sparse/expert-style student automatically won inside this harness. It did not.

## Turning Teacher Traces Into SFT Rows

The teacher data step is where agent distillation becomes concrete.

![Teacher trajectory to SFT rows](assets/teacher-trajectory-to-sft-sketchnote.png)

The GPT teacher was run on the 879 train tasks. It solved 446 of them:

```text
teacher train success: 446/879 = 50.7%
submitted: 879/879
parse failures: 0
repeated-action failures: 0
```

Only successful trajectories were trusted. Those 446 successful tasks became 1046 canonical SFT rows, because each successful task can contribute multiple next-action examples.

The final frozen SFT data for the CUDA runs:

| Data property | Value |
| --- | ---: |
| Source SFT rows | 1046 |
| Rows fitting 4096 tokens | 1042 |
| Train rows | 990 |
| Validation rows | 52 |
| Max sequence length | 4096 |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Learning rate | 5e-5 |

The target is canonical JSON. I do not train the student on whatever messy wrapper text the teacher happened to emit. The harness needs one action, so the training target is one action:

```json
{"action":"submit_sql","sql":["SELECT * FROM track WHERE track_id = (SELECT MAX(track_id) FROM track);"]}
```

That little detail is important. For agent distillation, the target is not "the teacher sounded right." The target is "the teacher chose an action that the harness can execute."

## A Concrete Held-Out Task

Here is the kind of task the model sees on eval:

```text
Database: chinook

User issue:
I want to find the latest track_id and use that id to filter records
in the track table.

Buggy SQL:
WITH vars AS (SELECT COUNT(*) AS vars_id FROM track)
SELECT * FROM track WHERE track_id = vars_id
```

The bug is subtle but common: `COUNT(*)` is not the latest id. The intended shape is to use `MAX(track_id)`.

The base 0.8B student usually did not even get to that level of SQL repair. It often inspected the schema, then inspected it again, then stopped as a repeated action. That is a harness-control failure.

The tuned 0.8B student handled this example much better. It inspected the schema, ran a small query to check the latest id, and submitted SQL using `MAX(track_id)`. That is exactly the kind of behavior I wanted distillation to transfer.

But a story about one task can fool you. The aggregate eval is what matters.

## Results

All rows below are from the same 220-task held-out eval split and the same fixed SQL harness.

![Held-out success chart](assets/current-eval-success-chart.png)

| Run | Success | Submitted | Parse Stops | Repeat Stops | Max/Runtime Stops |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B base | 1/220 = 0.5% | 1 | 11 | 208 | 0 |
| Qwen3.5-0.8B SFT, 1046 rows | 44/220 = 20.0% | 204 | 6 | 10 | 0 |
| Qwen3.5-0.8B SFT, submit rows duplicated once | 38/220 = 17.3% | 199 | 9 | 11 | 1 |
| Qwen3.5-2B base | 0/220 = 0.0% | 1 | 0 | 219 | 0 |
| Qwen3.5-2B SFT, 1046 rows | 57/220 = 25.9% | 206 | 5 | 9 | 0 |
| LFM2.5-8B-A1B SFT, 1046 rows | 47/220 = 21.4% | 186 | 7 | 27 | 0 |
| Qwen3.5-35B-A3B 8-bit local teacher candidate | 96/220 = 43.6% | 202 | 0 | 9 | 9 |
| GPT 5.5 medium teacher | 115/220 = 52.3% | 220 | 0 | 0 | 0 |

The first visible result is that SFT really did teach harness behavior.

The base Qwen students almost never submitted. The 0.8B base submitted once. The 2B base submitted once. After distillation, both submitted on most tasks: 204/220 for the 0.8B student and 206/220 for the 2B student.

That is not a cosmetic change. A model that does not submit cannot pass hidden tests.

![Harness behavior chart](assets/harness-behavior-chart.png)

The second result is that harness control is not the same as SQL competence.

The tuned students submitted often, but many submissions were wrong. Qwen3.5-2B was the strongest student in this round at 57/220. That is a big jump from the base model, but still far below the GPT teacher at 115/220 and below the local Qwen35 teacher candidate at 96/220.

The third result is a useful negative result. After normal fine-tuning, I tried duplicating the final `submit_sql` rows once. The hope was simple: maybe the small model needed more weight on final-answer behavior.

It got worse:

```text
0.8B SFT:              44/220 = 20.0%
0.8B SFT submit x2:    38/220 = 17.3%
```

It lost 17 tasks the normal 0.8B adapter solved, gained 11 new solved tasks, and increased SQL execution errors. So the issue was not simply "make it submit more." The student had already learned to submit. The harder part was choosing better SQL before submitting.

## Reading The Qwen35 Result

I also ran `mlx-community/Qwen3.5-35B-A3B-8bit` locally on the whole current eval split.

It landed at 96/220, or 43.6%. That makes it much stronger than the tuned students, but still below the GPT 5.5 medium teacher in this harness.

The behavior was also different. Qwen35 submitted 202 times, had zero parse failures, and still hit 7 max-turn stops, 9 repeated-action stops, and 2 runtime errors. In one long failure it generated a large self-debate about an ambiguous SQL prompt until the response hit the length limit. That is not a benchmark problem. It is part of the model behavior being measured by this fixed harness.

The important comparison is:

```text
Qwen3.5-2B SFT:         57/220 = 25.9%
Qwen3.5-35B-A3B 8-bit:  96/220 = 43.6%
GPT 5.5 medium:        115/220 = 52.3%
```

So the student has learned part of the teacher behavior, but not enough to look like a compressed teacher yet.

## What I Think Happened

This round mostly transferred **interaction format** and **basic tool-use rhythm**.

The base models were stuck at the door. They could emit something parseable sometimes, but they repeated actions and almost never reached final evaluation. SFT moved them into the room: inspect, query, observe, submit.

The remaining gap is SQL decision quality. The student has to infer the correct repair from the user issue, schema, observations, and sometimes misleading buggy SQL. The SFT rows teach the shape of teacher actions, but a small model can still choose the wrong join, the wrong aggregation, the wrong filter, or the wrong interpretation of the user request.

That is why the result is both encouraging and humbling:

```text
base 0.8B -> tuned 0.8B: 0.5% to 20.0%
base 2B   -> tuned 2B:   0.0% to 25.9%
teacher ceiling here:    52.3% for GPT 5.5 medium
```

Distillation clearly changed the behavior. It did not close the teacher gap.

## What This Post Does Not Claim

This is not a claim that the benchmark is solved. It is not a claim that validation loss is enough. It is not a claim that a larger or sparse model automatically wins.

It is also not a moving-target eval. The harness, split, hidden tests, and scoring stayed fixed for these comparisons. That is the main reason the negative result is useful: when submit-row duplication made the 0.8B model worse, I could interpret it as a training-data/optimization result, not as an environment change.

The exact commands, paths, adapters, and mirrored eval JSONs are in the README and repo artifacts. This post is the map: what kind of distillation this is, how the agent loop is built, how teacher traces become SFT rows, and what the fixed harness measured.

## Takeaway

The first version of the pipeline works:

```text
teacher runs fixed harness
  -> keep successful trajectories
  -> turn each next action into SFT rows
  -> train a smaller student
  -> rerun the same hidden-test eval
```

For a 0.8B SQL tool-use agent, that was enough to turn a model that almost never submitted into one that completed most harness runs and solved 20% of the held-out tasks. The 2B student reached 25.9%. A local Qwen35 teacher candidate reached 43.6%. GPT 5.5 medium reached 52.3%.

That is the honest shape of the result: real transfer, real gap, fixed measurement.
