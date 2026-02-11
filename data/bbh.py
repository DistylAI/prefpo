"""BBH (BIG-Bench-Hard) convenience data loader."""

import random
from typing import Any, Literal

import datasets

from prefpo.types import Sample

MULTIPLE_CHOICE_TASKS: set[str] = {
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
}

BINARY_TASKS: dict[str, set[str]] = {
    "formal_fallacies": {"valid", "invalid"},
    "boolean_expressions": {"true", "false"},
    "causal_judgement": {"yes", "no"},
    "navigate": {"yes", "no"},
    "sports_understanding": {"yes", "no"},
    "web_of_lies": {"yes", "no"},
}

EXACT_MATCH_TASKS: dict[str, str] = {
    "object_counting": "natural",
    "multistep_arithmetic_two": "integer",
    "word_sorting": "words",
}


def task_type(subset_name: str) -> Literal["multiple_choice", "binary", "exact_match"]:
    """Return the task type for a BBH subset."""
    if subset_name in MULTIPLE_CHOICE_TASKS:
        return "multiple_choice"
    if subset_name in BINARY_TASKS:
        return "binary"
    if subset_name in EXACT_MATCH_TASKS:
        return "exact_match"
    raise ValueError(
        f"Unknown BBH subset: {subset_name}. "
        f"Known: {sorted(MULTIPLE_CHOICE_TASKS | set(BINARY_TASKS) | set(EXACT_MATCH_TASKS))}"
    )


def _convert_record(record: dict[str, Any], index: int, subset_name: str) -> Sample:
    """Convert a raw HuggingFace record to a Sample."""
    question = record["question"]
    if not question.endswith("\n"):
        question = question + "\n"

    target = str(record["target"]).strip()

    # For MC tasks, include choices in the question text
    if subset_name in MULTIPLE_CHOICE_TASKS:
        choices_text_list = record.get("choices", {}).get("text", [])
        labels_list = record.get("choices", {}).get("label", [])
        if choices_text_list and labels_list:
            choices_str = "\n".join(
                f"{labels_list[i]} {choices_text_list[i]}"
                for i in range(len(choices_text_list))
            )
            question = f"{question}\n{choices_str}"
        target = target.upper()

    return Sample(index=index, question=question, target=target)


def load_bbh(
    subset_name: str,
    train_size: int,
    val_size: int,
    test_size: int | None = None,
    seed: int | None = None,
) -> tuple[list[Sample], list[Sample], list[Sample] | None]:
    """Load a BBH subset and split into train/val/test lists of Sample.

    Args:
        subset_name: BBH task name (e.g. "disambiguation_qa").
        train_size: Number of training samples.
        val_size: Number of validation samples.
        test_size: Number of test samples (None = all remaining).
        seed: Random seed for shuffling.

    Returns:
        (train, val, test) where test is None if test_size=0 and val_size=0.
    """
    # Validate task name
    task_type(subset_name)

    ds = datasets.load_dataset(
        "Joschka/big_bench_hard",
        name=subset_name,
        split=subset_name,
    )

    total = len(ds)
    min_required = train_size + val_size + (test_size or 0)
    if min_required > total:
        raise ValueError(
            f"Requested {min_required} samples (train={train_size}, val={val_size}, "
            f"test={test_size}) but {subset_name} only has {total}"
        )

    all_indices = list(range(total))

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(all_indices)
    else:
        random.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]

    if test_size is None:
        test_indices = all_indices[train_size + val_size :]
    elif test_size == 0:
        test_indices = []
    else:
        test_indices = all_indices[
            train_size + val_size : train_size + val_size + test_size
        ]

    train = [_convert_record(ds[i], i, subset_name) for i in train_indices]
    val = [_convert_record(ds[i], i, subset_name) for i in val_indices]
    test = [_convert_record(ds[i], i, subset_name) for i in test_indices] if test_indices else None

    return train, val, test
