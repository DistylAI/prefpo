"""IFEval-Hard dataset loader — a curated 148-sample subset of IFEval.

IFEval-Hard (rahul-singhal/IFEval-Hard) filters the original 541 IFEval
samples down to 148 that GPT-4o scores below 100% on, making it a harder
and more discriminative benchmark for prompt optimization.

Dataset: https://huggingface.co/datasets/rahul-singhal/IFEval-Hard
"""

from typing import Any

from datasets import load_dataset

from prefpo.config import PrefPOConfig
from prefpo.grading.ifeval import IFEvalGrader, get_human_readable_criteria

DATASET_ID = "rahul-singhal/IFEval-Hard"
DATASET_SIZE = 148

_dataset = None


def _get_dataset():
    """Get or load the cached IFEval-Hard dataset."""
    global _dataset
    if _dataset is None:
        _dataset = load_dataset(DATASET_ID, split="train")
    return _dataset


def _convert_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw HuggingFace record to a sample dict with criteria."""
    criteria = []
    for i, inst_id in enumerate(sample["instruction_id_list"]):
        kwargs = sample["kwargs"][i]
        criteria.append(get_human_readable_criteria(inst_id, kwargs))

    return {
        "key": sample["key"],
        "prompt": sample["prompt"],
        "instruction_id_list": sample["instruction_id_list"],
        "kwargs": sample["kwargs"],
        "criteria": criteria,
    }


def load_ifeval_hard_sample(idx: int) -> dict[str, Any]:
    """Load a single IFEval-Hard sample by index.

    Returns:
        dict with keys: key, prompt, instruction_id_list, kwargs, criteria
    """
    ds = _get_dataset()
    return _convert_sample(ds[idx])


def load_ifeval_hard_dataset() -> list[dict[str, Any]]:
    """Load the full IFEval-Hard dataset (148 samples).

    Returns:
        List of dicts, each with: key, prompt, instruction_id_list, kwargs, criteria
    """
    ds = _get_dataset()
    return [_convert_sample(ds[idx]) for idx in range(len(ds))]


def build_ifeval_hard_config(
    sample: dict[str, Any],
    base_config: PrefPOConfig | None = None,
    n_eval_trials: int = 20,
) -> tuple[PrefPOConfig, IFEvalGrader]:
    """Build a per-sample PrefPOConfig and IFEvalGrader.

    Args:
        sample: Dict from load_ifeval_hard_sample()
        base_config: Optional base config to override defaults
        n_eval_trials: Number of evaluation trials for the grader

    Returns:
        (config, grader) tuple ready for optimize()
    """
    if base_config is not None:
        config = base_config.model_copy(deep=True)
        config.mode = "standalone"
        config.pool.initial_prompts = [sample["prompt"]]
        config.pool.prompt_role = "user"
        config.discriminator.show_expected = True
        config.discriminator.criteria = sample["criteria"]
    else:
        config = PrefPOConfig(
            mode="standalone",
            task_model={"name": "openai/gpt-4o"},
            discriminator={
                "show_expected": True,
                "criteria": sample["criteria"],
            },
            pool={
                "initial_prompts": [sample["prompt"]],
                "prompt_role": "user",
            },
        )

    grader = IFEvalGrader(
        instruction_id_list=sample["instruction_id_list"],
        kwargs=sample["kwargs"],
        n_eval_trials=n_eval_trials,
    )

    return config, grader
