# PrefPO: Preference-based Prompt Optimization

Lightweight, async preference-based prompt optimization. Give PrefPO a pool of prompt candidates and a grader — it iteratively improves them using LLM-as-judge feedback.

- **Two optimization modes**: instruction (shared prompt + data samples) and standalone (prompt-is-the-task, no samples needed)
- **Low-data**: works with small datasets that don't need to be labeled; standalone mode needs no dataset
- **Text evaluation signal**: discriminator criteria are plain text, so you can start optimizing quickly
- **Minimal surface area**: one entry point (`optimize()`), one abstraction to implement (`Grader`)
- **Fast**: all LLM calls run concurrently via `asyncio.gather` with a shared semaphore

## How It Works

PrefPO implements the PRPO (Preference-based Prompt Optimization) loop:

1. **Sample** two prompts from a pool
2. **Generate** outputs from both using the task model
3. **Discriminate** — an LLM judge picks the better prompt and explains why
4. **Optimize** — a second LLM call improves the losing prompt using the judge's feedback
5. **Evaluate** the new prompt with your grader, add it to the pool, repeat
6. **Select** — after all iterations, return the prompt with the highest grader score

The discriminator and optimizer share context via message-passing and use [litellm](https://docs.litellm.ai/docs/providers) for model routing.

## Installation

```bash
pip install -e .
```

For IFEval/IFBench support (requires the official IFEval checker):

```bash
pip install -e ".[ifeval]"
```

Requires Python 3.11+. Set the API key for your provider as an environment variable (e.g. `OPENAI_API_KEY`). Model names use litellm's `"provider/model"` format.

> **Model defaults and compatibility:**
>
> | Role | Default | Purpose |
> |------|---------|---------|
> | Task model | *none — you choose* | The model being optimized (generates responses) |
> | Discriminator | `openai/gpt-5` (reasoning) | Compares prompt pairs and picks the better one |
> | Optimizer | `openai/gpt-5` (reasoning) | Rewrites the losing prompt using judge feedback |
>
> The discriminator and optimizer require structured JSON output via JSON schema mode. OpenAI, Anthropic (Claude 3.5+), and Gemini (2.0+) all enforce this natively and are fully supported. Other providers may vary — check [litellm's structured output docs](https://docs.litellm.ai/docs/completion/json_mode) for the full list. The task model has no such restriction — it can be any litellm-supported provider.

## Quick Start — Instruction Mode

Instruction mode optimizes a shared instruction that gets prepended to every question in a dataset (either as the user prompt or the system prompt). Use this when you have a single prompt you want to optimize for a dataset of many questions. Each prompt is scored by your grader on the validation set (or train if no val is provided), and the highest-scoring prompt is returned. You can use a built-in grader or write your own.

**With a built-in grader** — PrefPO includes graders for BBH tasks that check model outputs against ground truth answers:

```python
from prefpo import PrefPOConfig, optimize
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader

train, val, test = load_bbh("disambiguation_qa", train_size=50, val_size=50, seed=42)
grader = get_bbh_grader("disambiguation_qa")

config = PrefPOConfig(
    mode="instruction",
    task_model={"name": "openai/gpt-4o"},
    pool={"initial_prompts": [
        "Answer the question. End with ANSWER: <letter>.",
        "Think step by step. End with ANSWER: <letter>.",
    ]},
    run={"iterations": 10, "output_dir": "results/bbh"},
)

result = optimize(config, grader=grader, train=train, val=val, test=test)
print(f"Best score: {result.best_score:.3f}")
print(f"Best prompt: {result.best_prompt.value}")
```

**With a custom grader** — if you want custom evaluation logic (e.g. LLM-as-judge, partial credit, or domain-specific checks), write your own grader. It receives the prompt and data samples, generates outputs, and scores them however you want:

```python
from prefpo import PrefPOConfig, Grader, GradeResult, optimize
from prefpo.generate import generate_outputs
from prefpo.types import Sample

class LLMJudgeGrader(Grader):
    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        score = 0
        for o, s in zip(outputs, samples):
            # Use call_llm to ask an LLM to judge the response
            from prefpo import call_llm
            judge_resp = await call_llm(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": f"Question: {s.question}\nExpected: {s.target}\nResponse: {o.response}\n\nScore 0 or 1:"}],
            )
            score += int("1" in judge_resp.output_text)
        return GradeResult(score=score / len(outputs), n=len(outputs))

train = [Sample(index=0, question="What is 2+2?", target="4"),
         Sample(index=1, question="What is 3*5?", target="15")]

config = PrefPOConfig(
    mode="instruction",
    task_model={"name": "openai/gpt-4o"},
    pool={"initial_prompts": ["Solve the math problem.", "Think step by step."]},
    run={"iterations": 5},
)

result = optimize(config, grader=LLMJudgeGrader(), train=train)
```

## Quick Start — Standalone Mode

Standalone mode optimizes prompts where the prompt itself is the task (e.g. "Write a 300+ word summary on..." or "Create a blog post about..."). No dataset is needed — just a prompt, a grader, and criteria are enough to run optimization. Your grader scores each prompt directly by generating outputs and evaluating them, and the highest-scoring prompt is returned. Use this for tasks like instruction-following (IFEval) where each prompt has its own success criteria. Note: your grader can also be an LLM-judge when the evaluation criteria are nuanced and can't be checked programmatically.

```python
from prefpo import PrefPOConfig, Grader, GradeResult, optimize
from prefpo.generate import generate_standalone

class MyGrader(Grader):
    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_standalone(prompt, model_config, semaphore, n=3)
        required_keywords = ["renewable energy", "carbon emissions", "solar", "wind"]
        checks_passed = 0
        for o in outputs:
            text = o.response.lower()
            has_length = len(o.response.split()) < 200
            has_keywords = all(kw in text for kw in required_keywords)
            checks_passed += (has_length + has_keywords) / 2
        return GradeResult(score=checks_passed / len(outputs), n=len(outputs))

config = PrefPOConfig(
    mode="standalone",
    task_model={"name": "openai/gpt-4o"},
    discriminator={"criteria": ["response must be less than 200 words",
                                "response must include the keywords: 'renewable energy', 'carbon emissions', 'solar', 'wind'"]},
    pool={"initial_prompts": ["Write a blog post about the benefits of clean energy that is less than 200 words."]},
    run={"iterations": 5},
)

result = optimize(config, grader=MyGrader())
```

A single initial prompt is fine — PrefPO will automatically generate a variant to seed the pool with two candidates.

## Examples

The [`examples/`](examples/) directory contains full, runnable scripts:

| File | Mode | What it does |
|------|------|-------------|
| [`gsm8k.py`](examples/gsm8k.py) | instruction | Custom data loader + custom grader for GSM8K math |
| [`bbh.py`](examples/bbh.py) | instruction | Built-in BBH data loader + auto-selected grader |
| [`ifbench.py`](examples/ifbench.py) | standalone | Custom grader with IFBench constraint checkers |
| [`ifeval.py`](examples/ifeval.py) | standalone | Built-in IFEval loader + grader (541 samples) |
| [`ifeval_hard.py`](examples/ifeval_hard.py) | standalone | Built-in IFEval-Hard loader (148 curated samples) |
| [`judges.py`](examples/judges.py) | — | Run hack detection and hygiene scoring on sample prompts |
| [`ifeval_hard_llm_judge.py`](examples/ifeval_hard_llm_judge.py) | standalone | IFEval-Hard with an LLM judge grader instead of programmatic checks |

`gsm8k.py` and `ifbench.py` are heavily commented and walk through every config option. The other files show how to use built-in data loaders and graders for quick setup. YAML config examples are also included (`bbh_config.yaml`, `ifeval_config.yaml`).

```bash
# Instruction mode — built-in BBH data + grader
python examples/bbh.py

# Standalone mode — custom grader with constraint checkers
python examples/ifbench.py

# Prompt quality judges — hack detection + hygiene scoring
python examples/judges.py

# Standalone mode — LLM judge grader (no programmatic checks)
python examples/ifeval_hard_llm_judge.py
```

### Integration tests

Short scripts that verify different model/provider combinations work end-to-end:

| File | What it tests |
|------|--------------|
| [`test_multitrial.py`](examples/test_multitrial.py) | Multi-trial (`n_trials=2`), `vary_seed`, non-reasoning OpenAI discriminator (`gpt-4.1`) |
| [`test_anthropic_disc.py`](examples/test_anthropic_disc.py) | Anthropic discriminator (Claude Sonnet 4.5), `prompt_role="system"`, `update_strategy="replace"` |
| [`test_anthropic_opt.py`](examples/test_anthropic_opt.py) | Anthropic reasoning optimizer (Claude Sonnet 4.5), standalone mode |
| [`test_all_anthropic.py`](examples/test_all_anthropic.py) | All-Anthropic pipeline — no OpenAI key needed |

## YAML Configuration

Configs can be defined in YAML and loaded with `PrefPOConfig.from_yaml()`:

```yaml
mode: "instruction"

task_model:
  name: "openai/gpt-4o"
  temperature: 0.0

discriminator:
  model:
    name: "openai/gpt-5"
    is_reasoning: true
    reasoning_effort: "medium"
  show_expected: true
  criteria: "correctness and quality of reasoning"

optimizer:
  model:
    name: "openai/gpt-5"
    is_reasoning: true
    reasoning_effort: "medium"
  constraints: "Do not remove answer formatting rules (e.g. 'ANSWER: $LETTER')."

pool:
  initial_prompts:
    - "Answer the question. End with ANSWER: <letter>."
    - "Think step by step. End with ANSWER: <letter>."
  prompt_role: "user"
  update_strategy: "add"
  sampling_seed: 42

run:
  iterations: 15
  max_concurrent: 100
  output_dir: "results/bbh"
  save_outputs: true
  verbose: true
```

```python
config = PrefPOConfig.from_yaml("config.yaml")
```

## Key Concepts

### Criteria and Constraints

**Discriminator criteria** tell the judge what to evaluate on. **Constraints** provide additional rules for both the discriminator and optimizer. Both accept a string or list of strings.
Think of criteria as the evaluation signal for the judge — plain text is enough to get started.

```python
discriminator=DiscriminatorConfig(
    criteria=["correctness", "reasoning quality"],
    constraints="Ignore formatting differences",
)
optimizer=OptimizerConfig(
    constraints="Do not remove the 'ANSWER: $LETTER' format requirement",
)
```

Criteria appear in the discriminator prompt as `CRITERIA TO EVALUATE ON`. Optimizer constraints appear as `CONSTRAINTS FOR YOUR OUTPUT`.

### Expected Answers (`show_expected`)

When `show_expected=True`, the discriminator sees expected answers alongside model outputs. In instruction mode, this shows `Sample.target` for each question. In standalone mode, it calls `grader.check_output()` to annotate each output. This is a separate, lightweight method from `grade()` — it takes a single output string and returns a dict (e.g. `{"passed": True}`). The base class returns `None` by default, so you only need to implement it if you use `show_expected=True` in standalone mode.

### Prompt Role

`prompt_role` controls how the prompt is sent to the task model:

- **`"user"`** (default): the instruction is prepended to the question in a single user message
- **`"system"`**: the instruction is sent as a separate system message, with the question as the user message

Standalone mode requires `"user"` since the prompt is the entire user input.

### Single-Prompt Pools

PrefPO needs at least two prompts for pairwise comparison. If you provide only one initial prompt, PrefPO automatically generates a variant using the optimizer model (default: `openai/gpt-5`). The variant rewrites the prompt while preserving the same task and constraints — it appears in the pool as `variant_0`. This means you can start optimization with a single prompt:

```python
pool={"initial_prompts": ["Write a blog post about clean energy under 200 words."]}
```

The variant generator uses the discriminator's `criteria` to inform the rewrite, so setting good criteria helps produce a useful starting variant. If you want full control over both starting prompts, provide two explicitly.

### Pool Strategies

- **`"add"`** (default): the improved prompt is added to the pool, which grows over iterations. Good for exploration.
- **`"replace"`**: the improved prompt replaces the losing prompt, keeping the pool at a fixed size. We found this to be much less effective than `"add"` and should most likely be avoided.

### Multi-Trial

Set `n_trials > 1` to run independent optimization trials in parallel and pick the best:

```python
config = PrefPOConfig(
    ...,
    run={"iterations": 10, "n_trials": 3, "vary_seed": True},
)
```

With `vary_seed=True`, each trial gets a different sampling seed (`sampling_seed + trial_index`). Results are aggregated with mean/std statistics and a summary is written to the output directory.

### Batch Optimization

For standalone tasks like IFEval where each sample is optimized independently, `run_ifeval_batch()` runs multiple samples concurrently:

```python
from prefpo.ifeval_batch import run_ifeval_batch

results = await run_ifeval_batch(
    sample_indices=[0, 5, 10],
    n_eval_trials=20,
    batch_size=3,    # max concurrent samples
    dataset="ifeval_hard",  # or "ifeval" (default)
)
print(f"Success rate: {results['aggregate_metrics']['success_rate']:.1%}")
```

The batch runner writes per-sample results to separate subdirectories and produces a `batch_summary.json` with aggregate metrics. Individual sample failures are caught and logged without crashing the batch. See [`examples/ifeval_hard.py`](examples/ifeval_hard.py) for a full working example.

## Built-in Datasets

PrefPO includes data loaders for three datasets. Each loader returns samples ready for `optimize()`.

### BBH (BIG-Bench-Hard)

27 reasoning tasks from the [BBH](https://huggingface.co/datasets/Joschka/big_bench_hard) dataset across three answer types: multiple choice (18 tasks), binary (6), and exact match (3).

```python
from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader

train, val, test = load_bbh("disambiguation_qa", train_size=50, val_size=50, seed=42)
grader = get_bbh_grader("disambiguation_qa")  # auto-selects MultipleChoiceGrader
```

`get_bbh_grader()` picks the right grader based on the task type — no need to specify it manually.

### IFEval

541 instruction-following samples from [google/IFEval](https://huggingface.co/datasets/google/IFEval). Each sample has its own set of formatting constraints (e.g. "respond in all caps", "include exactly 3 bullet points").

```python
from prefpo.data.ifeval import load_ifeval_sample, build_ifeval_config

sample = load_ifeval_sample(0)
config, grader = build_ifeval_config(sample, n_eval_trials=20)
result = optimize(config, grader=grader)
```

`build_ifeval_config()` creates a standalone config and `IFEvalGrader` in one call, with human-readable criteria auto-generated from the constraint specs.

### IFEval-Hard

A curated 148-sample subset of IFEval ([rahul-singhal/IFEval-Hard](https://huggingface.co/datasets/rahul-singhal/IFEval-Hard)) — samples where GPT-4o scores below 100%, making it a harder and more discriminative benchmark.

```python
from prefpo.data.ifeval_hard import load_ifeval_hard_sample, build_ifeval_hard_config

sample = load_ifeval_hard_sample(0)
config, grader = build_ifeval_hard_config(sample, n_eval_trials=20)
result = optimize(config, grader=grader)
```

Same API as standard IFEval.

## Judges

PrefPO includes two standalone LLM judges for assessing prompt quality. These are useful for evaluating optimized prompts or as quality gates in your optimization pipeline.

### Prompt Hack Detection

Detects whether a prompt adds unnecessarily restrictive constraints beyond what the criteria require — e.g. tightening "under 200 words" to "under 150 words" to game the evaluation.

```python
import asyncio
from prefpo.judges import judge_prompt_hack

result = asyncio.run(judge_prompt_hack(
    prompt="Write a poem about the ocean. Keep it under 50 words.",
    criteria=["Response must have less than 100 words"],
))
print(result["grade"])      # "pass" or "fail"
print(result["reasoning"])  # explanation of the judgment
```

### Prompt Hygiene Scoring

Evaluates prompt quality on three axes (0-2 each): readability, specification quality, and maintainability.

```python
import asyncio
from prefpo.judges import judge_prompt_hygiene

result = asyncio.run(judge_prompt_hygiene(
    prompt="Write a blog post about Japan. Keep it under 300 words.",
    context="Response must have less than 300 words",  # optional
))
print(f"Readability: {result['readability_score']}/2")
print(f"Spec Quality: {result['spec_quality_score']}/2")
print(f"Maintainability: {result['maintainability_score']}/2")
print(result["overall_reasoning"])
```

When `context` is provided, the judge won't penalize for instructions required by that context — it only evaluates how well the constraints are expressed. When `context` is omitted, the judge evaluates the prompt in isolation.

Both judges use `openai/gpt-4.1` by default and accept a `model` parameter for any litellm-supported provider. See [`examples/judges.py`](examples/judges.py) for a full runnable demo.

## Custom Graders

Implement the `Grader` abstract class to plug in any evaluation logic. `grade()` owns the full pipeline — generate responses and evaluate them. Built-in graders are available for BBH tasks (`get_bbh_grader()`) and IFEval (`IFEvalGrader`).

**Instruction mode** — your grader receives the prompt and data samples. Use `generate_outputs()` to run the task model on every sample, then check the results:

```python
from prefpo import Grader, GradeResult
from prefpo.generate import generate_outputs

class MyInstructionGrader(Grader):
    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        correct = sum(1 for o, s in zip(outputs, samples)
                      if s.target.lower() in o.response.lower())
        return GradeResult(score=correct / len(outputs), n=len(outputs))
```

**Standalone mode** — `samples` is `None`. Use `generate_standalone()` to run the prompt directly and score the outputs:

```python
from prefpo import Grader, GradeResult
from prefpo.generate import generate_standalone

class MyStandaloneGrader(Grader):
    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_standalone(prompt, model_config, semaphore, n=3)
        score = sum(1 for o in outputs if len(o.response.split()) < 200) / len(outputs)
        return GradeResult(score=score, n=len(outputs))

    def check_output(self, output, prompt_text=None):
        """Optional: annotate individual outputs for standalone trajectories.
        Called when show_expected=True. Return a dict or None."""
        return {"passed": len(output.split()) < 200}
```

**LLM-as-judge grader** — use `call_llm` or `call_llm_json` inside `grade()` when you want an LLM to evaluate responses instead of programmatic checks:

```python
from prefpo import Grader, GradeResult, call_llm
from prefpo.generate import generate_outputs

class LLMJudgeGrader(Grader):
    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        score = 0
        for o, s in zip(outputs, samples):
            resp = await call_llm(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": f"Question: {s.question}\nExpected: {s.target}\nResponse: {o.response}\n\nScore 0 or 1:"}],
            )
            score += int("1" in resp.output_text)
        return GradeResult(score=score / len(outputs), n=len(outputs))
```

`call_llm` returns a litellm response object (with `.output_text`). For structured JSON output, use `call_llm_json` with a JSON schema — see [`examples/ifeval_hard_llm_judge.py`](examples/ifeval_hard_llm_judge.py) for a full example.

## API Reference

All public symbols are exported from the top-level `prefpo` package.

| Symbol | Description |
|--------|-------------|
| `optimize(config, grader, ...)` | Run optimization (sync wrapper) |
| `optimize_async(config, grader, ...)` | Run optimization (async) |
| `optimize_multi_trial(config, grader, ...)` | Run multiple independent trials in parallel |
| `PrefPOConfig` | Top-level configuration (mode, models, pool, run settings) |
| `ModelConfig` | Model name, temperature, reasoning settings |
| `DiscriminatorConfig` | Judge model, criteria, constraints, show_expected |
| `OptimizerConfig` | Optimizer model and constraints |
| `OptimizationResult` | Result with best_prompt, best_score, history, total_tokens |
| `MultiTrialResult` | Aggregated results across trials (mean, std, per-trial data) |
| `Prompt` | A prompt being optimized (value, role, name, metadata) |
| `PromptRole` | Enum: `USER` or `SYSTEM` |
| `Sample` | A single evaluation sample (question, target, metadata) |
| `Grader` | Abstract base class — implement `grade()` for custom evaluation |
| `GradeResult` | Grading result (score, n, per_sample, outputs) |
| `generate_outputs(...)` | Run the task model on a list of samples (instruction mode) |
| `generate_standalone(...)` | Run the task model N times on a single prompt (standalone mode) |
| `call_llm(...)` | Low-level LLM call via litellm — useful in custom graders |
| `judge_prompt_hack(...)` | Detect if a prompt over-restricts beyond its criteria |
| `judge_prompt_hygiene(...)` | Score prompt readability, spec quality, and maintainability (0-2 each) |

## CLI

```bash
python -m prefpo \
  --config config.yaml \
  --dataset bbh \
  --subset disambiguation_qa \
  --train-size 50 \
  --val-size 50 \
  --test-size 100 \
  --seed 42 \
  -v
```

## Output

Each run writes to `{output_dir}/run_{timestamp}_{id}/`:

| File | Contents |
|------|----------|
| `config.json` | Full configuration snapshot |
| `iteration_history.jsonl` | Per-iteration details: prompts, scores, feedback, token usage |
| `final_pool_state.json` | All prompts with their scores at the end of the run |
| `summary.json` | Best prompt, best score, total token usage |

Set `save_outputs=True` to additionally write `grading_outputs.jsonl` — the raw model responses from every grading call. Useful for debugging grader behavior and understanding score changes.

The `OptimizationResult` returned by `optimize()` contains `best_prompt`, `best_score`, `best_test_score`, `final_pool`, `history`, and `total_tokens`.

### Token Tracking

Token usage is tracked separately for the discriminator (judge) and optimizer LLM calls:

```python
result = optimize(config, grader=grader, train=train, val=val)

# Aggregated totals across all iterations
print(result.total_tokens)
# {"discriminator": {"input_tokens": 15230, "output_tokens": 4210, "total_tokens": 19440},
#  "optimizer":     {"input_tokens": 12100, "output_tokens": 3850, "total_tokens": 15950}}

# Per-iteration breakdown available in history
for record in result.history:
    print(f"Iter {record.iteration}: disc={record.discriminator_usage['total_tokens']}, "
          f"opt={record.optimizer_usage['total_tokens']}")
```

Token counts are also saved to `summary.json` (totals) and `iteration_history.jsonl` (per-iteration). Note: task model tokens used during grading are not tracked — only the discriminator and optimizer calls.

## Citation

If you use PrefPO in your research, please cite:

```bibtex
@article{prefpo2026,
  title={TODO},
  author={TODO},
  year={2026},
}
```
