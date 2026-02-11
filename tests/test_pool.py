"""Tests for prefpo.pool — PromptPool."""

import pytest
from prefpo.pool import PromptPool
from prefpo.types import Prompt, PromptRole
def _make_prompt(value: str, name: str = "") -> Prompt:
    return Prompt(value=value, role=PromptRole.USER, name=name)
def test_auto_naming():
    pool = PromptPool([_make_prompt("a"), _make_prompt("b")])
    assert pool.entries[0].name == "seed_0"
    assert pool.entries[1].name == "seed_1"
def test_preserves_existing_names():
    pool = PromptPool([_make_prompt("a", name="custom"), _make_prompt("b")])
    assert pool.entries[0].name == "custom"
    assert pool.entries[1].name == "seed_1"
def test_sample_pair():
    pool = PromptPool([_make_prompt("a"), _make_prompt("b")])
    p1, p2 = pool.sample_pair()
    assert p1 != p2
    assert p1 in pool.entries
    assert p2 in pool.entries
def test_sample_pair_deterministic():
    pool1 = PromptPool([_make_prompt("a"), _make_prompt("b"), _make_prompt("c")], seed=42)
    pool2 = PromptPool([_make_prompt("a"), _make_prompt("b"), _make_prompt("c")], seed=42)
    for _ in range(5):
        a1, b1 = pool1.sample_pair()
        a2, b2 = pool2.sample_pair()
        assert a1.value == a2.value
        assert b1.value == b2.value
def test_add():
    pool = PromptPool([_make_prompt("a"), _make_prompt("b")])
    assert len(pool.entries) == 2
    pool.add(_make_prompt("c", name="new"))
    assert len(pool.entries) == 3
    assert pool.entries[-1].name == "new"
def test_replace_non_preferred():
    pool = PromptPool([_make_prompt("a"), _make_prompt("b")])
    new = _make_prompt("c", name="improved")
    pool.replace_non_preferred(new, "seed_0", "seed_1", preferred=1)
    names = [e.name for e in pool.entries]
    assert "seed_0" in names
    assert "improved" in names
    assert "seed_1" not in names
def test_score_cache_by_value():
    pool = PromptPool([_make_prompt("same text"), _make_prompt("other")])
    pool.set_score(pool.entries[0], 0.8)
    # Same text, different name
    dup = _make_prompt("same text", name="dup")
    assert pool.get_score(dup) == 0.8
def test_get_score_none_if_unscored():
    pool = PromptPool([_make_prompt("a"), _make_prompt("b")])
    assert pool.get_score(pool.entries[0]) is None
def test_best():
    pool = PromptPool([_make_prompt("a"), _make_prompt("b"), _make_prompt("c")])
    pool.set_score(pool.entries[0], 0.5)
    pool.set_score(pool.entries[1], 0.9)
    pool.set_score(pool.entries[2], 0.7)
    assert pool.best().value == "b"
def test_best_unscored_returns_first():
    pool = PromptPool([_make_prompt("a"), _make_prompt("b")])
    assert pool.best().value == "a"
def test_to_dict():
    pool = PromptPool([_make_prompt("a")])
    pool.set_score(pool.entries[0], 0.5)
    d = pool.to_dict()
    assert len(d["prompts"]) == 1
    assert d["prompts"][0]["value"] == "a"
    assert d["prompts"][0]["score"] == 0.5
def test_empty_pool_rejected():
    with pytest.raises(ValueError):
        PromptPool([])
def test_single_prompt_pair_rejected():
    pool = PromptPool([_make_prompt("a")])
    with pytest.raises(ValueError):
        pool.sample_pair()
