"""Tests for dataset loading — verifies all three datasets can be retrieved.

These tests download from HuggingFace (cached after first run).
Run with: pytest tests/test_datasets.py -v
"""

import pytest

# --- BBH ---
def test_bbh_loads():
    from prefpo.data.bbh import load_bbh

    train, val, test = load_bbh("disambiguation_qa", train_size=3, val_size=2, test_size=2, seed=42)
    assert len(train) == 3
    assert len(val) == 2
    assert len(test) == 2
    for s in train:
        assert s.question
        assert s.target
# --- IFEval ---
def test_ifeval_loads():
    from prefpo.data.ifeval import load_ifeval_sample, load_ifeval_dataset

    sample = load_ifeval_sample(0)
    assert "prompt" in sample
    assert "criteria" in sample
    assert "instruction_id_list" in sample
    assert "kwargs" in sample
    assert len(sample["criteria"]) > 0
    assert len(sample["instruction_id_list"]) == len(sample["criteria"])
def test_ifeval_dataset_size():
    from prefpo.data.ifeval import load_ifeval_dataset

    ds = load_ifeval_dataset()
    assert len(ds) == 541
# --- IFEval-Hard ---
def test_ifeval_hard_loads():
    from prefpo.data.ifeval_hard import load_ifeval_hard_sample, load_ifeval_hard_dataset

    sample = load_ifeval_hard_sample(0)
    assert "prompt" in sample
    assert "criteria" in sample
    assert "instruction_id_list" in sample
    assert "kwargs" in sample
    assert len(sample["criteria"]) > 0
    assert len(sample["instruction_id_list"]) == len(sample["criteria"])
def test_ifeval_hard_dataset_size():
    from prefpo.data.ifeval_hard import load_ifeval_hard_dataset

    ds = load_ifeval_hard_dataset()
    assert len(ds) == 148
def test_ifeval_hard_criteria_are_human_readable():
    """Criteria should be descriptive strings, not raw instruction IDs."""
    from prefpo.data.ifeval_hard import load_ifeval_hard_sample

    sample = load_ifeval_hard_sample(0)
    for criterion in sample["criteria"]:
        assert isinstance(criterion, str)
        assert len(criterion) > 10  # not just an ID like "keywords:existence"
        assert ":" not in criterion[:20] or "must" in criterion.lower()  # looks like a sentence
