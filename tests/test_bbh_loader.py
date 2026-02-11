"""Tests for prefpo.data.bbh — BBH dataset loader."""

from prefpo.data.bbh import (
    BINARY_TASKS,
    EXACT_MATCH_TASKS,
    MULTIPLE_CHOICE_TASKS,
    load_bbh,
    task_type,
)
def test_task_type_mc():
    assert task_type("disambiguation_qa") == "multiple_choice"
    assert task_type("hyperbaton") == "multiple_choice"
def test_task_type_binary():
    assert task_type("navigate") == "binary"
    assert task_type("boolean_expressions") == "binary"
def test_task_type_exact():
    assert task_type("object_counting") == "exact_match"
    assert task_type("word_sorting") == "exact_match"
def test_task_type_unknown():
    import pytest
    with pytest.raises(ValueError):
        task_type("nonexistent_task")
def test_task_registries_disjoint():
    mc = MULTIPLE_CHOICE_TASKS
    binary = set(BINARY_TASKS)
    exact = set(EXACT_MATCH_TASKS)
    assert mc.isdisjoint(binary)
    assert mc.isdisjoint(exact)
    assert binary.isdisjoint(exact)
def test_load_bbh_splits():
    train, val, test = load_bbh("disambiguation_qa", train_size=3, val_size=2, test_size=5, seed=42)
    assert len(train) == 3
    assert len(val) == 2
    assert len(test) == 5
def test_load_bbh_deterministic():
    t1, v1, _ = load_bbh("disambiguation_qa", train_size=3, val_size=2, seed=42)
    t2, v2, _ = load_bbh("disambiguation_qa", train_size=3, val_size=2, seed=42)
    assert [s.index for s in t1] == [s.index for s in t2]
    assert [s.index for s in v1] == [s.index for s in v2]
def test_load_bbh_samples_have_targets():
    train, _, _ = load_bbh("disambiguation_qa", train_size=3, val_size=2, seed=42)
    for s in train:
        assert s.target != ""
        assert s.question != ""
def test_load_bbh_no_test():
    train, val, test = load_bbh("disambiguation_qa", train_size=3, val_size=2, test_size=0, seed=42)
    assert len(train) == 3
    assert len(val) == 2
    assert test is None
# --- Edge case tests ---

import pytest
def test_load_bbh_bounds_check():
    """Requesting more samples than available raises ValueError."""
    with pytest.raises(ValueError, match="only has"):
        load_bbh("disambiguation_qa", train_size=9999, val_size=9999, seed=42)
def test_load_bbh_seed_none():
    """seed=None still returns valid splits (non-deterministic)."""
    train, val, _ = load_bbh("disambiguation_qa", train_size=3, val_size=2, seed=None)
    assert len(train) == 3
    assert len(val) == 2
    for s in train:
        assert s.question != ""
        assert s.target != ""
