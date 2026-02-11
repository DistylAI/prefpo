"""IFBench standalone-mode example — optimizes prompts to follow constraints.

Standalone mode optimizes the prompt itself as one self-contained unit — there
is no separate dataset of questions. This is useful when the prompt IS the task
(e.g. "write a poem with exactly 5 sentences" or "respond using only lowercase").

The loop works like this:
  1. Sample two prompt variants from the pool
  2. Generate outputs from each by sending them to the task model (gpt-4o)
  3. A judge (gpt-5) compares the outputs and picks a winner + gives feedback
  4. An optimizer (gpt-5) rewrites the loser using the feedback
  5. Score the improved prompt (here: run it 5 times, check constraint compliance)
  6. Add it to the pool and repeat

With a single initial prompt, PrefPO auto-generates a variant using the
optimizer model so the first iteration has a pair to compare.

Key differences from instruction mode (see gsm8k.py):
  - No train/val/test data — the grader handles everything internally
  - The grader ignores the `samples` argument and uses its own evaluation logic
  - check_output() enables show_target, so the judge sees pass/fail annotations
  - prompt_role must be "user" (standalone doesn't support system prompts)
"""

import evaluation_lib
import instructions_registry
from datasets import load_dataset

from prefpo import PrefPOConfig, Grader, GradeResult, optimize
from prefpo.generate import generate_standalone


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_ifbench_dataset = None


def _get_dataset():
    global _ifbench_dataset
    if _ifbench_dataset is None:
        _ifbench_dataset = load_dataset("allenai/IFBench_test", split="train")
    return _ifbench_dataset


def load_ifbench_sample(idx: int) -> dict:
    """Load a single IFBench sample by index.

    Returns dict with: key, prompt, instruction_id_list, kwargs, criteria.
    """
    ds = _get_dataset()
    sample = ds[idx]

    # Generate human-readable criteria from constraint specs
    criteria = []
    for i, inst_id in enumerate(sample["instruction_id_list"]):
        cls = instructions_registry.INSTRUCTION_DICT[inst_id]
        checker = cls(inst_id)
        kw = {k: v for k, v in sample["kwargs"][i].items() if v is not None}
        kw = {k: int(v) if isinstance(v, float) and v == int(v) else v
              for k, v in kw.items()}
        desc = checker.build_description(**kw)
        criteria.append(desc)

    return {
        "key": sample["key"],
        "prompt": sample["prompt"],
        "instruction_id_list": sample["instruction_id_list"],
        "kwargs": sample["kwargs"],
        "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class IFBenchGrader(Grader):
    """Custom grader that ignores the samples argument entirely.

    Instead of comparing against labeled data, this grader:
      1. Generates N outputs from the prompt using generate_standalone()
      2. Runs each output through IFBench's official constraint checkers
      3. Returns score = fraction of outputs that pass ALL constraints

    This pattern works for any task where you have a programmatic or LLM-based
    way to evaluate quality without needing labeled examples.
    """

    def __init__(self, instruction_id_list: list[str], kwargs: list[dict],
                 original_prompt: str, n_eval_trials: int = 5):
        self.instruction_id_list = instruction_id_list
        self.kwargs = kwargs
        self.original_prompt = original_prompt
        self.n_eval_trials = n_eval_trials

    async def grade(self, prompt, samples, model_config, semaphore):
        # `samples` is ignored — we generate outputs and score them ourselves
        outputs = await generate_standalone(
            prompt, model_config, semaphore, n=self.n_eval_trials,
        )

        all_pass = 0
        per_sample = []
        for o in outputs:
            inp = evaluation_lib.InputExample(
                key=0,
                instruction_id_list=self.instruction_id_list,
                prompt=self.original_prompt,
                kwargs=self.kwargs,
            )
            result = evaluation_lib.test_instruction_following_strict(
                inp, {self.original_prompt: o.response},
            )
            if result.follow_all_instructions:
                all_pass += 1
            per_sample.append({
                "all_pass": result.follow_all_instructions,
                "per_constraint": dict(zip(
                    self.instruction_id_list, result.follow_instruction_list,
                )),
            })

        score = all_pass / len(outputs) if outputs else 0.0
        raw_outputs = [{"sample_index": i, "response": o.response} for i, o in enumerate(outputs)]
        return GradeResult(score=score, n=len(outputs), per_sample=per_sample, outputs=raw_outputs)

    def check_output(self, output: str, prompt_text: str | None = None) -> dict:
        """Annotate a single output for show_target trajectory display.

        When show_target=True, PrefPO calls this on each generated output so the
        judge can see pass/fail annotations alongside the text. Only needed if
        you set show_target=True in the discriminator config.
        """
        inp = evaluation_lib.InputExample(
            key=0,
            instruction_id_list=self.instruction_id_list,
            prompt=self.original_prompt,
            kwargs=self.kwargs,
        )
        result = evaluation_lib.test_instruction_following_strict(
            inp, {self.original_prompt: output},
        )
        return {
            "all_pass": result.follow_all_instructions,
            "per_constraint": dict(zip(
                self.instruction_id_list, result.follow_instruction_list,
            )),
        }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # IFBench has 300 samples total. Each is optimized independently, so
    # running the full dataset takes a while. Here we just optimize one sample
    # as a demo — add more indices to optimize more (e.g. [5, 22, 65]).
    sample_indices = [22]
    results = []

    for i, idx in enumerate(sample_indices):
        sample = load_ifbench_sample(idx)
        print(f"\n{'='*60}")
        print(f"Sample {idx} (key={sample['key']})")
        print(f"Constraints: {sample['instruction_id_list']}")
        for c in sample["criteria"]:
            print(f"  - {c}")
        print(f"Prompt: {sample['prompt'][:150]}...")

        grader = IFBenchGrader(
            instruction_id_list=sample["instruction_id_list"],
            kwargs=sample["kwargs"],
            original_prompt=sample["prompt"],
            # More trials = more reliable scores but more API calls per iteration
            n_eval_trials=5,
        )

        config = PrefPOConfig(
            # "standalone" = the prompt itself is the task (no dataset).
            # For dataset-based tasks, use "instruction" instead (see gsm8k.py).
            mode="standalone",

            task_model={"name": "openai/gpt-4o"},

            discriminator={
                # show_target=True requires check_output() on the grader so the judge
                # can see pass/fail annotations. Helps the judge make better decisions
                # when you have a programmatic way to check correctness.
                "show_target": True,
                # criteria: passed as human-readable descriptions so the judge knows
                # what constraints to look for
                "criteria": sample["criteria"],
            },

            pool={
                # Single prompt — PrefPO will auto-generate a variant using the
                # optimizer model so the first iteration has a pair to compare.
                "initial_prompts": [sample["prompt"]],
                # Standalone mode requires "user" (system prompts not supported)
                "prompt_role": "user",
            },

            run={
                "iterations": 3,
                "output_dir": f"results/ifbench/sample_{idx}",
                # save_outputs=True logs all model responses to grading_outputs.jsonl
                "save_outputs": True,
            },
        )

        # No train/val/test needed — the grader handles scoring internally
        result = optimize(config, grader=grader)
        results.append((idx, sample, result))

        print(f"\nResult: score={result.best_score:.3f}")
        print(f"Best prompt:\n{result.best_prompt.value[:200]}...")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for idx, sample, result in results:
        print(f"  Sample {idx}: {result.best_score:.3f}  constraints={sample['instruction_id_list']}")
