# Distillation Blog Series Plan

## Purpose

This project is a practical notebook-blog series about training a very small open model to become a better tool-calling, agentic model through distillation.

The positioning is:

> I can take a tiny open model, train it on a realistic tool-use environment, and show which distillation methods actually improve reliability, cost, and agent behavior.

This should not become a generic survey of every knowledge-distillation paper. The story should stay grounded in one repeatable benchmark, one small student model, one strong teacher model, and a small number of realistic training methods.

The posts should be written as notebook tutorials. The reader should learn the concepts by running the benchmark loop, inspecting traces, training the model, and comparing behavior before and after.

## Core Research Question

Can `mlx-community/Qwen3.5-0.8B-MLX-bf16` learn reliable multi-turn retail tool use from a stronger same-family teacher, `Qwen/Qwen3.5-35B-A3B`, using practical distillation methods?

The series should compare these regimes on the same focused task family:

1. Baseline prompting only.
2. Sequence-level SFT on successful teacher trajectories.
3. Soft-label or logit distillation on the same trajectories.
4. On-policy distillation, where the student generates trajectories and the teacher scores the states the student actually visits.
5. Optional RL, where the benchmark simulator/evaluator supplies reward.

The intended claim is not that one method always wins. The likely story is:

> SFT teaches the small model the tool-use policy and action format. Logit distillation transfers richer token-level preferences. On-policy distillation teaches recovery from the student's own mistakes. RL can use the simulator directly, but sparse rewards make it a later step, not the first training signal.

## Current Benchmark Decision

### Primary Benchmark

Use **τ³-Bench retail** from Sierra's `sierra-research/tau2-bench` repository:

```text
sierra-research/tau2-bench
domain: retail
revision: c42db6cc223ef37c02ef2fb2f605ae0a4ca9afd6
```

Cached files for Blog 1:

```text
data/raw/tau2/retail/tasks.json
data/raw/tau2/retail/split_tasks.json
data/raw/tau2/retail/policy.md
data/raw/tau2/retail/db.json
data/raw/tau2/retail/tools.py
```

Current split from the notebook:

```text
114 retail tasks total
74 train tasks
40 held-out test tasks
16 retail tools
```

Use the official train/test split. Do not force the old habit of 50 eval tasks onto this benchmark.

### Naming Convention

Use these names consistently:

- **τ-Bench**: the benchmark family.
- **τ³-Bench retail**: the current benchmark release/domain used for this blog.
- `sierra-research/tau2-bench`: the upstream GitHub repository that contains the current τ³-Bench code/data.
- `tau2`: the Python package, CLI command, and internal data-path prefix used by that repository.

So prose should say **τ³-Bench retail** when discussing the benchmark. Code, paths, and runtime imports may say `tau2` because that is the actual package/repository interface.

### Why τ³-Bench Retail

For the first blog, we want a focused specialization story:

> Can a 0.8B model become better at one realistic customer-support workflow?

τ³-Bench retail is a good fit because it has one domain, one policy, one hidden retail database, one set of tools, a simulated user, and an evaluator. That lets the post focus on specialization instead of mixing unrelated APIs and task styles.

## Benchmark Mental Model

τ³-Bench retail should be taught as an environment, not just a dataset.

The model sees:

```text
retail policy
conversation history
available tool schemas
tool observations from previous tool calls
```

The model does not see:

```text
hidden user scenario
hidden database state
evaluation criteria
ground-truth reference actions
simulator internals
```

The user simulator sees the hidden user scenario and produces customer messages. The environment owns the hidden database and executes retail tools. The evaluator checks whether the final state and conversation satisfy the task.

For teaching and debugging, notebooks may render a visible debug user message from the hidden scenario. That must be labeled clearly as a debug probe, not official τ³-Bench scoring.

## Models

### Student

Use:

```text
mlx-community/Qwen3.5-0.8B-MLX-bf16
```

Why:

- It is tiny enough for fast iteration.
- It is a realistic edge/local model size.
- It has native chat and tool-call tokens.
- It is small enough that improvements from distillation should be visible.
- It is the bf16 MLX conversion of Qwen 0.8B, so the Apple Silicon runtime path is explicit and non-quantized.

### Teacher

Intended teacher family:

```text
Qwen/Qwen3.5-35B-A3B
```

Practical local teacher artifacts can include quantized checkpoints such as:

```text
mlx-community/Qwen3.5-35B-A3B-8bit
```

The exact teacher runtime must be recorded per run:

```text
model id
quantization
serving backend
temperature
top_p
top_k
context length
max output tokens
tool parser version
dataset revision
```

We already learned that the serving backend matters. MLX raw serving can be fast but fragile around chat/tool-call post-processing. Ollama is useful for quick generation comparisons, but it is not the best path for future logit or probability work. vLLM raw completions are attractive because the harness can own prompt rendering and Qwen XML parsing, and vLLM has a clearer path to logprobs.

Do not silently swap teacher runtime, checkpoint, context length, or decoding policy. If the experiment changes, the notebook should say so explicitly.

### Tokenizer Compatibility

Tokenizer compatibility between the bf16 MLX student and the 35B-A3B teacher has been verified from Hugging Face tokenizer artifacts:

```text
mlx-community/Qwen3.5-0.8B-MLX-bf16 tokenizer.json == Qwen/Qwen3.5-35B-A3B tokenizer.json
mlx-community/Qwen3.5-0.8B-MLX-bf16 vocab.json == Qwen/Qwen3.5-35B-A3B vocab.json
mlx-community/Qwen3.5-0.8B-MLX-bf16 merges.txt == Qwen/Qwen3.5-35B-A3B merges.txt
vocab_size = 248320
```

Important token IDs match:

```text
<tool_call>       248058
</tool_call>      248059
<tool_response>   248066
</tool_response>  248067
<think>           248068
</think>          248069
<|im_start|>      248045
<|im_end|>        248046
```

This means same-tokenizer logit KD, top-k KL, and on-policy distillation can compare student and teacher distributions directly.

Important caveat:

`tokenizer_config.json` is not byte-identical because chat-template behavior can differ around default `enable_thinking`. Experiments must render prompts consistently for both student and teacher.

Recommended first default:

```python
enable_thinking = False
```

This keeps tool-call outputs easier to parse and benchmark.

## Output Format

Use Qwen3.5's native chat template and tool-call format.

The tokenizer template has two layers:

1. Tool definitions are rendered as JSON schemas inside `<tools>...</tools>`.
2. Model tool-call outputs are requested in an XML-like format inside `<tool_call>...</tool_call>`.

The output tool-call format looks like:

```text
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
```

The harness should parse this format into structured calls, then execute those calls through the τ³-Bench retail environment.

Do not train with arbitrary one-off JSON formats unless an adapter is explicitly needed. Keep a single boundary:

```text
Qwen tool-call text <-> structured function call object <-> `tau2` runtime retail environment
```

## Evaluation Metrics

Use the official τ³-Bench environment and evaluator whenever possible.

Track at least:

```text
task success
final state correctness
policy compliance
function name validity
argument validity
tool-call syntax validity
number of tool calls
unnecessary tool calls
loop rate
stop-too-early rate
invalid output rate
latency / tokens generated
```

Use pass@1 as the main evaluation number. If we later run pass@N, report it as a separate metric, because pass@N answers a different question:

```text
pass@1: How reliable is one deployment attempt?
pass@N: Can sampling/search find a successful attempt if we can afford retries?
```

For blog posts, always show both:

1. Aggregate score table.
2. Concrete failure examples before and after training.

The practical reader should be able to see what changed behaviorally, not only numerically.

## Training Data Rules

Do not train on held-out test tasks.

Do not generate the first SFT dataset directly from the benchmark reference actions and call it distillation. That can be useful for debugging, but distillation means the teacher model generated the trajectory.

For sequence-level teacher distillation:

1. Run the teacher on train tasks only.
2. Let the teacher interact with the same harness, parser, tools, simulator, and scorer as the student.
3. Keep successful full-task trajectories.
4. Slice each successful trajectory into next-action SFT rows.

One solved task can produce many SFT rows:

```text
input  = conversation state before the teacher action
target = teacher's next assistant action
```

If a task has multiple user turns, rows from later turns include the previous user turns, previous assistant actions, and previous tool observations. The student is trained to imitate the next decision from the full state it would actually see.

## Methods To Include

### 1. Baseline Prompting

Run `mlx-community/Qwen3.5-0.8B-MLX-bf16` on τ³-Bench retail without training.

Capture failure modes:

```text
invalid tool-call format
wrong tool
missing required argument
wrong argument value
failure to use tool result
failure to ask required confirmation
premature final answer
loops
policy violation
state mutation mistakes
```

This establishes the need for training.

### 2. Sequence-Level Teacher SFT

Use the teacher `Qwen/Qwen3.5-35B-A3B` to generate full trajectories on the retail train split.

Pipeline:

1. Render τ³-Bench retail policy, conversation, and tool schemas with the fixed Qwen chat template.
2. Ask teacher for the next action.
3. Parse Qwen tool-call text.
4. Execute the tool call in the retail environment.
5. Append the tool result.
6. Continue until the simulator/evaluator finishes the task.
7. Keep successful trajectories.
8. Convert successful teacher actions into SFT rows.
9. Fine-tune the student with normal next-token supervised training.

This is sequence-level distillation. It uses token-level loss, but it does not require teacher logits.

Expected result:

Better tool format, better policy following, and better multi-step behavior than the base student.

### 3. Soft-Label / Logit Distillation

Use successful trajectories as fixed sequences. Run the teacher in scoring mode over:

```text
prompt + teacher action completion
```

Then train the student to match the teacher's token distributions on the completion tokens.

Because tokenizer IDs match, the loss can compare student and teacher distributions directly.

Implementation choices:

- Start with top-k teacher logprobs, not full-vocabulary logits.
- Use top-k KL or cross-entropy over teacher top-k tokens to reduce memory and payload.
- Use the same prompts and successful trajectories as Blog 1 so the comparison is controlled.
- Keep decoding settings separate from scoring settings. Teacher-forcing/scoring asks: what probability did the teacher assign to each already-known token?

Important teaching point:

Teacher scoring an existing sequence can be done as a forward pass over the prompt plus completion. Teacher generation requires autoregressive sampling. This is why logprob collection is a different job from generating trajectories.

Expected result:

Better token-level imitation than plain SFT, especially around tool tokens, argument formatting, and stop behavior.

### 4. On-Policy Distillation

Pipeline:

1. Student generates a trajectory in the τ³-Bench retail environment.
2. The environment executes tools and returns observations.
3. The resulting student trajectory is given to the teacher in scoring mode.
4. The teacher returns token-level logprobs or distributions for the student's visited states.
5. The student is trained to move closer to the teacher on those states.

Why it matters:

Offline SFT trains on clean teacher trajectories. At inference time, the student continues from its own imperfect prefixes. On-policy distillation reduces this mismatch by training on student-generated states.

Possible implementation:

Prefer a minimal MLX loop on Mac if we can keep the benchmark/runtime interface clean. Keep TRL `GKDTrainer` as a reference implementation, not the default local path:

```text
student rollout -> teacher logprobs -> KL loss -> optimizer step
```

Keep the first implementation same-tokenizer. Do not introduce GOLD, ULD, or cross-tokenizer methods in the main path.

Expected result:

Better recovery from student-specific bad prefixes and fewer cascading tool-use failures.

### 5. RL In The Simulator

τ³-Bench can be used as an RL environment because it has state, actions, transitions, observations, a user simulator, and evaluator/reward.

Use RL as a later comparator, not the first method.

Reward candidates:

```text
final task success
final state correctness
policy compliance
valid tool-call format
penalty for loops
penalty for unnecessary calls
```

The blog angle:

RL is attractive because the benchmark has executable rewards. It is also harder because rewards can be sparse and delayed. Distillation gives denser supervision earlier.

Train only on train tasks or generated train-like tasks. Keep held-out test tasks untouched for honest evaluation.

## Blog Series Outline

### Blog 1: Sequence Distillation On A Retail Agent

Folder:

```text
1-distilling-a-0-8b-tool-calling-agent/
```

Title idea:

```text
Distilling a 0.8B Retail Tool-Calling Agent
```

Target notebook flow after the τ³-Bench rewrite:

```text
01_student_eval.ipynb
02_teacher_eval.ipynb
03_teacher_train_trajectories.ipynb
04_train_student_sft_mlx_lm.ipynb
05_eval_sft_student.ipynb
```

Current status:

- `01_student_eval.ipynb` has been pivoted to τ³-Bench retail.
- `02_teacher_eval.ipynb` runs the vLLM teacher on the held-out τ³-Bench retail test split.
- `03_teacher_train_trajectories.ipynb` now targets the τ³-Bench retail train split and extracts MLX-LM chat SFT rows from successful teacher trajectories.
- `04_train_student_sft_mlx_lm.ipynb` replaces the old HF TRL/PEFT path with MLX-LM LoRA training on Apple Silicon.
- `05_eval_sft_student.ipynb` loads the MLX-LM adapter and reruns held-out τ³-Bench retail eval.

Content:

- Teach what a benchmark, harness, simulator, environment, and reward mean in tool use.
- Load and inspect τ³-Bench retail.
- Show that the model sees policy, tools, conversation, and observations, not hidden state.
- Render Qwen tool prompts.
- Run the base student on one step and one visible teaching/debug trajectory.
- Batch independent first-action probes for speed.
- Wire the student into the official τ³-Bench retail runner and report held-out test pass@1.
- Run teacher eval on the same held-out retail test split.
- Generate teacher trajectories on the retail train split only.
- Build SFT rows from successful teacher actions.
- Fine-tune the 0.8B student with MLX-LM LoRA.
- Re-run the held-out retail test tasks and compare before vs after.

Teaching point:

Sequence distillation trains the student to imitate the teacher's selected next action. It is still token-level training, but it is not logit distillation because the student does not see the teacher's full token probability distribution.

Deliverable:

Numbered notebooks with baseline metrics, teacher trajectory samples, fine-tuning code, post-training metrics, and qualitative before/after examples.

### Blog 2: Soft Labels From The Same Teacher

Title idea:

```text
Text Is Not All The Teacher Knows
```

Content:

- Explain teacher-forcing/scoring mode.
- Explain logits, logprobs, soft labels, and dark knowledge.
- Reuse the exact successful trajectories from Blog 1.
- Ask the teacher what probability it assigned to the teacher action tokens.
- Train with top-k teacher logprobs on the same rows.
- Compare plain SFT vs soft-label/logit KD.

Deliverable:

KD training notebook and comparison table against Blog 1.

### Blog 3: On-Policy Distillation

Title idea:

```text
Let The Student Fail, Then Let The Teacher Score It
```

Content:

- Explain train-inference mismatch.
- Let the student generate trajectories inside τ³-Bench retail.
- Score student-generated states/actions with the teacher.
- Train with GKD/on-policy loss.
- Compare to offline sequence KD and logit KD.

Deliverable:

On-policy run and analysis of recovery from student-specific mistakes.

### Blog 4: RL In A Tool-Use Simulator

Title idea:

```text
When The Benchmark Becomes The Environment
```

Content:

- Reframe τ³-Bench retail using RL language: state, action, transition, observation, reward, policy.
- Show how the user simulator, retail tools, hidden database, and evaluator become the environment.
- Train or compare with a verifier-reward method such as GRPO or another practical RL loop.
- Explain reward sparsity and credit assignment.
- Compare RL against sequence distillation, soft-label distillation, and on-policy distillation on the same held-out test split.

Deliverable:

RL/simulator notebook with reward design, a training run or controlled comparator, and an honest cost/reliability comparison.

### Blog 5: The Practical Recipe

Title idea:

```text
The Distillation Recipe I Would Actually Use
```

Content:

- Summarize all methods.
- Show final benchmark table.
- Discuss cost, complexity, and where each method helped.
- Recommend a practical sequence:

```text
baseline eval
sequence distillation on successful teacher traces
soft-label/logit KD when teacher logprobs are affordable
on-policy distillation when student-specific mistakes dominate
RL when the simulator reward is reliable enough to justify the complexity
```

Deliverable:

Final model, final benchmark, and honest lessons learned.

## Implementation Plan

### Step 1: Repository Layout

Use:

```text
docs/
1-distilling-a-0-8b-tool-calling-agent/
2-logit-distillation/
3-on-policy-distillation/
4-rl-in-a-tool-use-simulator/
5-practical-distillation-recipe/
data/raw/tau2/retail/
data/processed/
data/generated/
outputs/
outputs/local_traces/
```

Blog folders can contain multiple numbered notebooks when the workflow is too large for one notebook. Shared runtime code belongs in root `common/` so future blog posts reuse the same benchmark, tracing, teacher, and training helpers. Blog folders should contain notebooks and assets only; benchmark data belongs in root `data/`, and generated artifacts belong in root `outputs/`.

### Step 2: Environment

Use Python with:

```text
jupyter
ipykernel
transformers
datasets
accelerate
torch
mlx-lm
mlx-tune
`tau2` / `tau2-bench` runtime dependencies
```

Use `uv` for the repo environment.

For Apple Silicon teacher serving, keep candidates explicit:

```text
vLLM raw completions for generation/logprobs when available
MLX-LM raw serving for fast generation experiments
Ollama for quick generation comparisons, not logit-distillation work
```

### Step 3: Tokenizer Verification

Keep a notebook or script cell that verifies:

```text
tokenizer.json equality
vocab.json equality
merges.txt equality
important tool-token IDs
chat-template behavior under enable_thinking=False
```

This should run before logit-distillation experiments.

### Step 4: τ³-Bench Retail Data Loader

Load and cache:

```text
tasks.json
split_tasks.json
policy.md
db.json
tools.py
```

Create normalized examples:

```text
task_id
split
visible conversation state
available tools
hidden scenario metadata for simulator only
environment initialization data
evaluation metadata
```

### Step 5: Qwen Tool Parser

Parse Qwen tool-call text into structured calls:

```text
function_name
arguments
raw_text
parse_errors
```

Never use brittle keyword matching for correctness. Use structured parsing and environment/evaluator state checks.

### Step 6: Student Baseline Evaluator

Run `mlx-community/Qwen3.5-0.8B-MLX-bf16` on the held-out retail test split.

Store:

```text
raw model outputs
rendered prompts
parsed tool calls
tool execution logs
simulated user messages
final score
failure labels
local trace files
```

### Step 7: Teacher Eval

Run the teacher on the same held-out retail test split with the same harness and pass@1 decoding policy.

The teacher eval is not training data. It establishes whether the teacher is strong enough to be worth distilling.

### Step 8: Teacher Trajectory Collection

Run the teacher on retail train tasks only.

Keep:

```text
all attempts
successful full traces
SFT rows sliced from successful traces
failure examples
runtime config
```

Use pass@1 as the default collection policy unless a separate search experiment is explicitly being run.

### Step 9: Training Runs

Start with MLX-LM LoRA adapters for speed on Apple Silicon.

Evidence from the local smoke test:

- MLX-LM trained real τ³-shaped `messages + tools` rows with the non-quantized `mlx-community/Qwen3.5-0.8B-MLX-bf16` base, one LoRA layer, prompt masking, and about 4k token sequences. The smoke run completed with about 8.6 GB peak memory.
- A 4-bit `mlx-community/Qwen3.5-0.8B-4bit` base also trains, but it is a fallback for memory pressure, not the default student model for the blog.
- Earlier full-precision two-layer smoke runs hit Metal OOM on the same long tool/policy rows, so the first production config stays at one LoRA layer.
- MLX-Tune trained a smoke row and exposes useful future APIs such as GRPO/logprob helpers, but its saved adapter config did not faithfully reflect the one-layer LoRA smoke setting. Use MLX-LM CLI as the Blog 1 production training path until that is understood or fixed.

Suggested run order:

```text
run_00_student_baseline
run_01_teacher_eval
run_02_teacher_trajectory_sft
run_03_sft_student_eval
run_04_soft_label_kd
run_05_on_policy_gkd
run_06_rl_comparator_optional
```

### Step 10: Results Tracking

Every run should save:

```text
config
git commit if available
dataset revision/hash
model ids
serving backend
decoding config
adapter path
eval metrics
sample outputs
failure analysis
full local traces
```

Local JSON/JSONL trace logging should remain the debugging source of truth. MLflow can be used for dashboards, but local trace artifacts should be complete enough that we can debug without clicking through the UI.

## Risks And Decisions

### Official Simulator Integration

The official τ³-Bench simulator matters. For debug cells, a visible scripted user message is fine if clearly labeled. For benchmark claims, use the official runner/evaluator.

### Serving 35B-A3B

Even though 35B-A3B is sparse, the serving backend still needs to load a large checkpoint. Quantization, context length, and backend behavior can change quality.

Do not report failures caused by artificial context, output-token, or turn limits as model failures. If hardware forces a cap, record the cap and treat it as an experiment limitation.

### Retry And Sampling Policy

Main eval should be pass@1.

Retries, higher temperature, top-p changes, and pass@N are search strategies. They can be studied separately, but they should not be mixed into the main pass@1 deployment metric.

### Tool-Call Format Drift

The student may learn invalid variants of the XML-like tool-call format. Track syntax validity separately from task success.

### Thinking Tokens

The models include `<think>` and `</think>` tokens. For the first version, disable thinking in the chat template to simplify parsing and evaluation.

Later, run a separate ablation:

```text
enable_thinking=False vs enable_thinking=True
```

### Cross-Tokenizer Distillation

Do not include cross-tokenizer methods in the first series. Because the Qwen3.5 student and teacher share tokenizer artifacts, there is no need for ULD/GOLD initially.

Mention these only as future work.

### Avoid Overclaiming

The series should not claim to solve general agents. It should claim measurable improvement on a real stateful retail-support environment for a tiny model.

## Source Links

Core distillation posts and demos:

- Thinking Machines on-policy distillation: https://thinkingmachines.ai/blog/on-policy-distillation/
- Noah Ziems, Pedagogical RL: https://noahziems.com/pedagogical-rl
- SFT / RL / OPD distributional lens: https://nrehiew.github.io/blog/sft_rl_opd/
- Hugging Face GOLD / on-policy distillation: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
- Hugging Face TRL distillation trainer, reference path only: https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer

Models:

- Student MLX bf16: https://huggingface.co/mlx-community/Qwen3.5-0.8B-MLX-bf16
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5 collection: https://huggingface.co/collections/Qwen/qwen35
- MLX Qwen3.5-35B-A3B 8-bit: https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-8bit

Benchmarks:

- τ³-Bench implementation repository (`tau2-bench`; includes τ-Bench lineage): https://github.com/sierra-research/tau2-bench
- Original τ-Bench repository: https://github.com/sierra-research/tau-bench

Implementation references:

- MLX-LM repository and LoRA tooling: https://github.com/ml-explore/mlx-lm
- MLX-Tune LLM/GRPO tooling: https://arahim3.github.io/mlx-tune/llm.html
- TRL GKD trainer docs, reference path only: https://huggingface.co/docs/trl/gkd_trainer
- vLLM logprobs docs: https://docs.vllm.ai/en/stable/api/vllm/v1/engine/logprobs/
