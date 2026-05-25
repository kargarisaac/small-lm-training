# Distillation Blog Series Plan

## Purpose

This project is a practical blog series about training a very small model to become a better tool-calling, agentic model through distillation.

The positioning is:

> I can take a tiny open model, train it on a realistic tool-use task, and show which distillation methods actually improve reliability, cost, and agent behavior.

This should not become a generic survey of every knowledge-distillation paper. The story should stay grounded in one repeatable benchmark, one small student model, one strong teacher model, and a small number of realistic training methods.

The posts should be written as notebook tutorials. Each blog folder contains one main notebook with the educational explanation, runnable code, intermediate artifacts, and final result discussion in one place. The reader should learn the concepts by running the benchmark loop, training step, and comparison cells.

## Core Research Question

Can `Qwen/Qwen3.5-0.8B` learn reliable multi-turn tool calling from a stronger same-family teacher, `Qwen/Qwen3.5-35B-A3B`, using practical distillation methods?

The series should compare these training regimes on the same task:

1. Baseline prompting only.
2. Supervised fine-tuning on tool traces.
3. Synthetic teacher-trajectory distillation.
4. Best-of-N / verifier-filtered distillation.
5. Supervised logit distillation on successful trajectories.
6. On-policy distillation / GKD, where the student generates trajectories and the teacher scores them token-by-token.
7. Optional comparator: GRPO or another RL method using execution success as reward.

The intended claim is not that one method always wins. The likely story is:

> SFT teaches the format. Synthetic distillation teaches successful trajectories. Best-of-N improves data quality. Logit KD transfers richer token-level behavior. On-policy distillation teaches the student from the states it actually visits. RL can help at the end, but it is sparse and less convenient as the first method.

## Models

### Student

Use:

```text
Qwen/Qwen3.5-0.8B
```

Why:

- It is tiny enough for fast iteration.
- It is a realistic edge/local model size.
- The model card says Qwen3.5-0.8B is intended for prototyping, task-specific fine-tuning, and research/development.
- It has native chat/tool tokens in the tokenizer.
- It is small enough that improvements from distillation should be visible.

### Teacher

Use:

```text
Qwen/Qwen3.5-35B-A3B
```

Why:

- It is much stronger than the 0.8B student.
- It is a same-family Qwen3.5 model, which avoids cross-tokenizer distillation problems.
- It is an MoE model with roughly 3B active parameters per token, although the full 35B-class weights still need to be loaded for serving.
- It gives a clean story: distill agentic behavior from a sparse 35B teacher into a 0.8B student.

### Tokenizer Compatibility

Tokenizer compatibility has already been verified from Hugging Face tokenizer artifacts:

```text
Qwen/Qwen3.5-0.8B tokenizer.json == Qwen/Qwen3.5-35B-A3B tokenizer.json
Qwen/Qwen3.5-0.8B vocab.json == Qwen/Qwen3.5-35B-A3B vocab.json
Qwen/Qwen3.5-0.8B merges.txt == Qwen/Qwen3.5-35B-A3B merges.txt
```

Both text configs report:

```text
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

This means vanilla logit KD, top-k KL, GKD, and on-policy distillation can be used without Universal Logit Distillation, GOLD, or other cross-tokenizer alignment methods.

Important caveat:

`tokenizer_config.json` is not byte-identical because the chat template differs slightly around default `enable_thinking` behavior. The experiments must render prompts consistently for both student and teacher. Choose one mode and keep it fixed.

Recommended default:

```python
enable_thinking = False
```

This keeps tool-calling outputs easier to parse and benchmark.

## Task

Use a real multi-turn tool-calling benchmark, not a hand-made toy task.

Primary task:

```text
BFCL v3 multi-turn stateful tool use
```

Primary dataset:

```text
gorilla-llm/Berkeley-Function-Calling-Leaderboard
```

Start with these files/categories:

```text
BFCL_v3_multi_turn_base.json
BFCL_v3_multi_turn_composite.json
BFCL_v3_exec_simple.json
```

Start narrow:

1. First use `BFCL_v3_multi_turn_base.json`.
2. Once the pipeline works, add `BFCL_v3_multi_turn_composite.json`.
3. Use execution-based categories before broadening into all BFCL categories.

Why BFCL:

- It is a real public function-calling benchmark.
- It contains multi-turn and multi-step agentic interactions.
- It includes stateful API systems such as file system and ticket APIs.
- It supports execution-based scoring, not only string matching.
- It is close to the task we care about: a model must inspect state, choose tools, call them correctly, use tool outputs, and stop at the right time.

The benchmark should be framed as:

> Can a 0.8B model become a reliable action model inside an agent loop?

Do not frame the project as making a general chat model smarter. The model is being trained for a narrow but useful agentic capability: tool-use policy.

## Additional Training Datasets

Use these as warm-up or augmentation data, not as the final evaluation:

```text
Salesforce/xlam-function-calling-60k
lockon/ToolACE
```

Purpose:

- Teach basic function-calling format.
- Improve function selection and argument structure before stateful BFCL training.
- Provide a cheap SFT warm-up stage.

Keep final claims tied to BFCL evaluation, not only to these training datasets.

## Output Format

Use Qwen3.5's native chat template and tool-call format. The tokenizer template has two different layers:

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

The agent/eval harness should parse this format into structured calls.

Do not train with arbitrary one-off JSON formats unless the benchmark harness requires conversion. If conversion is needed, keep a single adapter layer:

```text
Qwen tool-call text <-> structured function call object <-> BFCL executor/scorer
```

## Evaluation Metrics

Use BFCL's official or repository-provided scoring whenever possible.

Track at least:

```text
task success / execution success
final state correctness
function name accuracy
argument accuracy
tool-call syntax validity
number of tool calls
unnecessary tool calls
loop rate
stop-too-early rate
invalid output rate
latency / tokens generated
```

For blog posts, always show both:

1. Aggregate score table.
2. Concrete failure examples before and after training.

The practical reader should be able to see what changed behaviorally, not only numerically.

## Methods To Include

Keep the series realistic. Do not start with cross-tokenizer distillation, multi-teacher distillation, or speculative decoding. These can be future posts.

### 1. Baseline Prompting

Run `Qwen/Qwen3.5-0.8B` on the selected BFCL split without training.

Capture failure modes:

- invalid tool-call format
- wrong function name
- missing required argument
- failure to use tool result
- loops
- early final answer
- state mutation mistakes

This post establishes the need for training.

### 2. SFT On Tool Traces

Train the student on real or derived tool-call traces using standard supervised fine-tuning.

Use:

```text
Salesforce/xlam-function-calling-60k
lockon/ToolACE
BFCL-derived gold traces if available or generated
```

Goal:

Teach syntax, tool schema following, and basic tool-selection behavior.

Expected result:

Large improvement in valid tool-call formatting and simple function selection, but weaker recovery on multi-turn stateful tasks.

### 3. Synthetic Teacher-Trajectory Distillation

Use the teacher `Qwen/Qwen3.5-35B-A3B` to generate full trajectories for BFCL-style prompts and tool schemas.

Pipeline:

1. Render BFCL prompt and tool schemas with the fixed Qwen chat template.
2. Ask teacher for next action.
3. Execute tool call in the BFCL environment.
4. Append tool result.
5. Continue until final answer or max step limit.
6. Score with BFCL/execution checker.
7. Keep successful trajectories.
8. SFT the student on successful teacher trajectories.

This is text-only sequence-level distillation. It does not require teacher logits.

Expected result:

Better multi-step behavior than generic SFT, because training data matches the benchmark environment and tool lifecycle.

### 4. Best-of-N / Verifier-Filtered Distillation

Generate multiple trajectories for each task, then keep the best ones.

Possible candidate generators:

```text
teacher only
student after SFT
mixture of teacher and student
```

Verifier:

Use BFCL execution success and final-state scoring.

Selection criteria:

1. Successful final state.
2. Valid tool-call syntax.
3. Fewer unnecessary calls.
4. No loops.
5. Shorter successful trajectory when equivalent.

This is the first post where the data pipeline becomes more important than the model architecture.

Expected result:

Higher quality training data and fewer learned bad habits.

### 5. Supervised Logit KD

Use successful trajectories as fixed sequences. Run the teacher over:

```text
prompt + tool trajectory + final answer
```

Then train the student to match teacher token distributions on the same tokens.

Because tokenizer IDs match, the loss can compare student and teacher distributions directly.

Implementation choices:

- Start with top-k teacher logprobs, not full-vocabulary logits.
- Use top-k KL or cross-entropy over teacher top-k tokens to reduce memory and payload.
- Use MLX-LM on Apple Silicon as the first local teacher runtime. Use the MLX server for generation and top-k logprobs, and direct MLX/Python calls for lower-level logits experiments if the server interface is too high-level.

Important teaching point:

Teacher scoring an existing length-N sequence can be done as a full forward/prefill pass over the sequence, while teacher generation requires N autoregressive decode steps. This is one reason logit-based scoring can be practical.

Expected result:

Better token-level imitation than plain SFT, especially around tool tokens, argument formatting, and stop behavior.

### 6. On-Policy Distillation / GKD

This is the main advanced post.

Pipeline:

1. Student generates the tool trajectory.
2. Execute tools as the trajectory unfolds.
3. Feed the resulting student trajectory to the teacher.
4. Teacher returns token-level logprobs/distributions on that trajectory.
5. Train the student to move closer to the teacher on states the student actually visited.

Why it matters:

Offline SFT and sequence KD train on clean teacher trajectories. At inference time, the student continues from its own mistakes. On-policy distillation reduces this train-inference mismatch by training on student-generated states.

Possible implementation:

Use TRL `GKDTrainer` or the newer TRL distillation trainer if it supports the needed Qwen3.5 architecture and teacher-server setup. If not, implement a minimal loop:

```text
student rollout -> teacher logprobs -> KL loss -> optimizer step
```

Keep the first implementation simple. Use same-tokenizer Qwen models. Do not introduce GOLD/ULD.

Expected result:

Better recovery from student-specific bad prefixes and fewer cascading tool-use failures.

### 7. Optional RL Comparator

Use GRPO or another verifier-based RL method as a comparator, not as the main story.

Reward:

```text
final-state success
valid tool-call format
correct function name
correct arguments
penalty for loops/unnecessary calls
```

The blog angle:

RL is attractive because BFCL has executable rewards, but sparse rewards can be inefficient for tiny models. Distillation gives denser supervision earlier.

## Blog Series Outline

### Blog 1: Sequence Distillation End To End

Folder:

```text
1-distilling-a-0-8b-tool-calling-agent/
```

Title idea:

```text
Distilling a 0.8B Tool-Calling Agent
```

Notebook:

```text
blog.ipynb
```

Content:

- Teach what a benchmark, simulator, and harness mean in the specific case of tool use.
- Load and inspect a small BFCL v3 multi-turn slice.
- Run the base `Qwen/Qwen3.5-0.8B` student on the same tasks.
- Parse Qwen native tool-call text and record execution or format failures.
- Generate teacher trajectories with `Qwen/Qwen3.5-35B-A3B` or a smaller temporary teacher if the 35B-A3B serving path is not available.
- Build a sequence-distillation dataset from successful teacher trajectories.
- Fine-tune the 0.8B student with normal next-token supervised training on the teacher's chosen trajectory text.
- Re-run the same benchmark slice and compare before vs after.

Teaching point:

Sequence distillation trains on the teacher's selected tokens. It is still a token-level training loss, but it is not logit distillation because the student does not see the teacher's full token probability distribution.

Deliverable:

One runnable notebook with baseline metrics, teacher trajectory samples, fine-tuning code, post-training metrics, and 3-5 qualitative before/after failure examples.

### Blog 2: Logit Distillation

Title idea:

```text
Text Is Not All The Teacher Knows
```

Content:

- Explain teacher logits and dark knowledge.
- Explain why same tokenizer matters.
- Show verified Qwen3.5 tokenizer compatibility.
- Train with top-k teacher logprobs on successful trajectories.
- Compare SFT vs logit KD on the same data.

Deliverable:

KD training run and comparison table.

### Blog 3: On-Policy Distillation

Title idea:

```text
Let The Student Fail, Then Let The Teacher Score It
```

Content:

- Explain train-inference mismatch.
- Student generates trajectories.
- Teacher scores student-generated trajectories token-by-token.
- Train with GKD/on-policy loss.
- Compare to offline KD.

Deliverable:

On-policy run and analysis of recovery from student-specific mistakes.

### Blog 4: RL In A Tool-Use Simulator

Title idea:

```text
When The Benchmark Becomes The Environment
```

Content:

- Reframe the BFCL harness using RL language: state, action, transition, observation, reward, policy.
- Build a small explicit simulator for one tool-use task before returning to BFCL.
- Train or compare with a verifier-reward method such as GRPO or another practical RL loop.
- Explain why sparse execution rewards are powerful but inconvenient as the first training signal.
- Compare RL against sequence distillation, logit distillation, and on-policy distillation on the same benchmark slice.

Deliverable:

RL/simulator notebook with a small environment, reward design, training run or controlled comparator, and an honest cost/reliability comparison.

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
Best-of-N filtering if data quality is the bottleneck
logit KD when teacher logprobs are affordable
on-policy distillation when student-specific mistakes dominate
RL when the simulator reward is reliable enough to justify the complexity
```

Deliverable:

Final model, final benchmark, and honest lessons learned.

## Implementation Plan

### Step 1: Repository Setup

Create or verify these folders:

```text
docs/
1-distilling-a-0-8b-tool-calling-agent/
2-logit-distillation/
3-on-policy-distillation/
4-rl-in-a-tool-use-simulator/
5-practical-distillation-recipe/
data/raw/
data/processed/
data/generated/
outputs/
```

Each blog folder should have one main `blog.ipynb`. Shared helper code can be introduced only after notebook duplication becomes painful. For Blog 1, prefer pure Python cells so the harness is visible before any framework abstraction.

### Step 2: Environment

Use Python with:

```text
jupyter
ipykernel
pandas
tqdm
transformers
datasets
trl
accelerate
peft
bitsandbytes or equivalent quantization support
mlx-lm
torch
pydantic
```

Use `uv` if the repo already uses it. Otherwise keep setup simple.

Blog 1 should start with the smallest useful stack: notebook tooling, `pydantic`, `torch`, `transformers`, `accelerate`, `pandas`, `tqdm`, and `mlx-lm` for the local teacher server. Add `trl`, `peft`, and quantization/fine-tuning libraries only when the fine-tuning cells need them.

### Step 3: Tokenizer Verification Cell

Add a notebook cell, and later a script if reuse becomes painful, that verifies:

```text
tokenizer.json equality
vocab.json equality
merges.txt equality
important tool-token IDs
chat-template behavior under enable_thinking=False
```

This should run before any logit-distillation experiment.

### Step 4: BFCL Data Loader

Load:

```text
gorilla-llm/Berkeley-Function-Calling-Leaderboard
```

Start with:

```text
BFCL_v3_multi_turn_base.json
```

Create normalized examples:

```text
task_id
messages
available_tools
initial_state
ground_truth/path if available
expected scoring metadata
```

### Step 5: Qwen Tool Parser

Parse Qwen tool-call text into structured calls:

```text
function_name
arguments
raw_text
parse_errors
```

Never use brittle keyword matching for correctness. Use structured parsing and executor state checks.

### Step 6: Baseline Evaluator

Run `Qwen/Qwen3.5-0.8B` on the chosen BFCL split.

Store:

```text
raw outputs
parsed tool calls
tool execution logs
final score
failure labels
```

### Step 7: SFT Dataset Builder

Convert xLAM/ToolACE/BFCL-derived examples into Qwen chat-template training sequences.

Keep the formatting consistent with the eval harness.

### Step 8: Teacher Trajectory Generator

Use `mlx-community/Qwen3.5-35B-A3B-4bit` through MLX-LM to generate trajectories. This is the practical local serving artifact for the same-family `Qwen/Qwen3.5-35B-A3B` teacher.

Start with a small batch and inspect outputs manually.

If serving 35B-A3B is too heavy, use `Qwen/Qwen3.5-9B` as a fallback for early pipeline testing, but keep 35B-A3B as the intended main teacher.

### Step 9: Training Runs

Start with LoRA/QLoRA-style adapters for speed.

Suggested run order:

```text
run_00_baseline
run_01_sft_warmup
run_02_synthetic_teacher_sft
run_03_best_of_n_sft
run_04_logit_kd
run_05_on_policy_gkd
run_06_rl_comparator_optional
```

### Step 10: Results Tracking

Every run should save:

```text
config
git commit if available
dataset version/hash
model ids
adapter path
eval metrics
sample outputs
failure analysis
```

## Risks And Decisions

### Serving 35B-A3B

Even though it is A3B active, the full MoE weights must be loaded. The default local serving path is a raw MLX server on Apple Silicon:

```bash
uv run python scripts/serve_teacher_mlx_raw.py
```

This server returns generated Qwen text directly. It intentionally avoids MLX-LM's OpenAI-compatible tool-call post-processing because that layer can hide Qwen tool-call text behind an empty `finish_reason="tool_calls"` response. The benchmark harness owns the Qwen XML parsing step.

If local serving is too expensive:

1. Use `Qwen/Qwen3.5-9B` to develop the pipeline.
2. Run the final teacher-generation/logprob jobs with 35B-A3B only where needed.

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

The series should not claim to solve general agents. It should claim measurable improvement on a real stateful tool-calling benchmark for a tiny model.

## Source Links

Core distillation posts and demos:

- Thinking Machines on-policy distillation: https://thinkingmachines.ai/blog/on-policy-distillation/
- Noah Ziems, Pedagogical RL: https://noahziems.com/pedagogical-rl
- SFT / RL / OPD distributional lens: https://nrehiew.github.io/blog/sft_rl_opd/
- Hugging Face GOLD / on-policy distillation: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
- Hugging Face TRL distillation trainer: https://huggingface.co/spaces/HuggingFaceTB/trl-distillation-trainer

Models:

- Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- Qwen3.5 collection: https://huggingface.co/collections/Qwen/qwen35

Datasets and benchmarks:

- Berkeley Function Calling Leaderboard dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- BFCL collection: https://huggingface.co/collections/gorilla-llm/berkeley-function-calling-leaderboard
- xLAM function-calling 60k: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- ToolACE: https://huggingface.co/datasets/lockon/ToolACE

Implementation references:

- TRL GKD trainer docs: https://huggingface.co/docs/trl/gkd_trainer
- vLLM logprobs docs: https://docs.vllm.ai/en/stable/api/vllm/v1/engine/logprobs/
