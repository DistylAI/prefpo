"""GSM8K instruction-mode example — optimizes a math prompt with PrefPO.

Instruction mode optimizes a shared instruction that gets prepended to every
question in a dataset. The loop works like this:

  1. Sample two prompts from the pool
  2. Run both on the training questions via the task model (gpt-4o)
  3. A judge (gpt-5) compares the outputs and picks a winner + gives feedback
  4. An optimizer (gpt-5) rewrites the loser using the feedback
  5. Score the improved prompt on the validation set
  6. Add it to the pool and repeat

After all iterations, the best prompt is evaluated on a held-out test set.

To adapt this to your own task:
  - Swap load_gsm8k() for your own data loader that returns list[Sample]
  - Subclass Grader and implement grade() with your scoring logic
  - Adjust the config options (see inline comments below)
"""

import re
import random

import datasets

from prefpo import PrefPOConfig, Grader, GradeResult, optimize
from prefpo.generate import generate_outputs
from prefpo.types import Sample


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k(train_size: int, val_size: int, test_size: int = 0,
               seed: int = 42) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Load GSM8K and split into train/val/test Sample lists."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")

    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    def to_sample(idx: int) -> Sample:
        record = ds[idx]
        answer_text = record["answer"]
        target = _normalize_number(answer_text.split("####")[-1].strip())
        return Sample(index=idx, question=record["question"], target=target)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size : train_size + val_size + test_size]

    train = [to_sample(i) for i in train_indices]
    val = [to_sample(i) for i in val_indices]
    test = [to_sample(i) for i in test_indices]
    return train, val, test


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def _normalize_number(s: str) -> str:
    s = s.strip().replace(",", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def _extract_answer(text: str) -> str | None:
    matches = list(re.finditer(r"ANSWER:\s*([^\n]+)", text, re.IGNORECASE))
    if matches:
        return _normalize_number(matches[-1].group(1))
    return None


class GSM8KGrader(Grader):
    """Exact-match grader: score = fraction of questions answered correctly."""

    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        sample_map = {s.index: s for s in samples}

        correct = 0
        per_sample = []
        for o in outputs:
            sample = sample_map[o.sample_index]
            pred = _extract_answer(o.response)
            is_correct = pred == sample.target if pred is not None else False
            if is_correct:
                correct += 1
            per_sample.append({
                "index": o.sample_index,
                "pred": pred,
                "target": sample.target,
                "correct": is_correct,
            })

        n = len(outputs)
        score = correct / n if n else 0.0
        return GradeResult(score=score, n=n, per_sample=per_sample)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train, val, test = load_gsm8k(train_size=30, val_size=30, test_size=30, seed=42)
    grader = GSM8KGrader()

    config = PrefPOConfig(
        # "instruction" = shared prompt + per-question data. "standalone" = prompt is the whole task.
        mode="instruction",

        # The model that answers questions. The judge/optimizer default to gpt-5.
        task_model={"name": "openai/gpt-4o"},

        discriminator={
            # show_target=True lets the judge see ground-truth answers alongside outputs.
            # Set False for subjective tasks with no single right answer.
            "show_target": True,
            # criteria: what the judge optimizes for (string or list of strings)
            "criteria": "correctness of the final numerical answer",
        },

        optimizer={
            # constraints: rules the optimizer must follow when rewriting prompts.
            # Useful for preserving format strings the grader depends on.
            "constraints": "Keep the formating instructions intact they are needed for evaluation.",
        },

        pool={
            # "user" = user message, "system" = system message. System can be more
            # authoritative but not all models handle it the same way.
            "prompt_role": "user",
            # Starting prompts. With one prompt, instruction mode will automatically generate a second prompt so the
            # first iteration has a pair to compare although it may not perserve the formatting instructions. We 
            # recommend making a variant manually (doesn't need to be that differnet)  or starting off with 2 of the same prompts.
            # We expect this doesn't affect final performance much, in practice. Two diverse seeds help explore more.
            "initial_prompts": [
                # "Solve the following math problem step by step. The last line of your response should be of the form 'ANSWER: $ANSWER' (without quotes) where $ANSWER is the answer to the problem.",
                "Think step by step to solve the following math problem. The last line of your response should be of the form 'ANSWER: $ANSWER' (without quotes) where $ANSWER is the answer to the problem.",
            ],
            # update_strategy: "add" (pool grows) or "replace" (fixed size, loser replaced).
            # "replace" keeps the pool tight for long runs.
        },

        run={
            "iterations": 3,
            "output_dir": "results/gsm8k",
            # save_outputs: True to log all model responses (useful for debugging, large files)
            # n_trials: run N independent trials and report mean/std
            # vary_seed: give each trial a different sampling seed
            # verbose: False to disable progress bar
        },
    )

    # train = shown to discriminator for trajectory generation
    # val   = used for scoring prompts each iteration (if None, falls back to train)
    # test  = held-out, only used to evaluate the final best prompt
    #
    # Note: the grader receives scoring samples but isn't forced to use them.
    # If you don't have labeled val data, you can write a grader that ignores
    # the samples argument and scores via its own logic (e.g. an LLM judge).
    # See ifbench.py for an example of a custom grader that doesn't use the samples argument.
    result = optimize(config, grader=grader, train=train, val=val, test=test)

    print(f"\nRun ID: {result.run_id}")
    print(f"Best score (val): {result.best_score:.3f}")
    print(f"Best score (test): {result.best_test_score:.3f}")
    print(f"Best prompt:\n{result.best_prompt.value}")
