"""IFEval dataset loader — loads from HuggingFace google/IFEval."""

from typing import Any

from datasets import load_dataset

from prefpo.config import PrefPOConfig
from prefpo.grading.ifeval import IFEvalGrader, get_human_readable_criteria

_ifeval_dataset = None


def _get_dataset():
    """Get or load the cached IFEval dataset."""
    global _ifeval_dataset
    if _ifeval_dataset is None:
        _ifeval_dataset = load_dataset("google/IFEval", split="train")
    return _ifeval_dataset


def load_ifeval_sample(idx: int) -> dict[str, Any]:
    """Load a single IFEval sample by index.

    Returns:
        dict with keys: key, prompt, instruction_id_list, kwargs, criteria
    """
    ds = _get_dataset()
    sample = ds[idx]

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


def load_ifeval_dataset() -> list[dict[str, Any]]:
    """Load the full IFEval dataset.

    Returns:
        List of dicts, each with: key, prompt, instruction_id_list, kwargs, criteria
    """
    ds = _get_dataset()
    samples = []
    for idx in range(len(ds)):
        samples.append(load_ifeval_sample(idx))
    return samples


def build_ifeval_config(
    sample: dict[str, Any],
    base_config: PrefPOConfig | None = None,
    n_eval_trials: int = 20,
) -> tuple[PrefPOConfig, IFEvalGrader]:
    """Build a per-sample PrefPOConfig and IFEvalGrader.

    Args:
        sample: Dict from load_ifeval_sample()
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
        config.discriminator.show_target = True
        config.discriminator.criteria = sample["criteria"]
    else:
        config = PrefPOConfig(
            mode="standalone",
            task_model={"name": "openai/gpt-4o"},
            discriminator={
                "show_target": True,
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
