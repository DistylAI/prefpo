"""Multi-trial test with non-reasoning OpenAI discriminator.

Exercises: multi-trial (n_trials=2), vary_seed, non-reasoning discriminator
(gpt-4.1), instruction mode with GSM8K custom grader.
"""

import re
import random

import datasets

from prefpo import PrefPOConfig, Grader, GradeResult, optimize
from prefpo.generate import generate_outputs
from prefpo.types import Sample


def _normalize(s: str) -> str:
    s = s.strip().replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


class GSM8KGrader(Grader):
    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        sample_map = {s.index: s for s in samples}
        correct = 0
        for o in outputs:
            s = sample_map[o.sample_index]
            matches = list(re.finditer(r"ANSWER:\s*([^\n]+)", o.response, re.IGNORECASE))
            pred = _normalize(matches[-1].group(1)) if matches else None
            if pred == s.target:
                correct += 1
        n = len(outputs)
        return GradeResult(score=correct / n if n else 0.0, n=n)


def load_gsm8k(train_size, val_size, seed=42):
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    def to_sample(idx):
        r = ds[idx]
        target = _normalize(r["answer"].split("####")[-1].strip())
        return Sample(index=idx, question=r["question"], target=target)

    train = [to_sample(i) for i in indices[:train_size]]
    val = [to_sample(i) for i in indices[train_size:train_size + val_size]]
    return train, val


if __name__ == "__main__":
    train, val = load_gsm8k(train_size=3, val_size=5)

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "openai/gpt-4o-mini"},
        discriminator={
            "model": {"name": "openai/gpt-4.1", "is_reasoning": False},
            "show_expected": True,
            "criteria": "correctness of the final numerical answer",
        },
        pool={
            "initial_prompts": [
                "Solve the following math problem step by step. The last line of your response should be of the form 'ANSWER: $ANSWER' (without quotes) where $ANSWER is the answer to the problem.",
            ],
        },
        run={
            "iterations": 2,
            "n_trials": 2,
            "vary_seed": True,
            "output_dir": "results/test_multitrial",
        },
    )

    result = optimize(config, grader=GSM8KGrader(), train=train, val=val)
    print(f"\nBest score: {result.best_score:.3f}")
    print(f"Best prompt:\n{result.best_prompt.value[:200]}")
