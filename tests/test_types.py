"""Tests for prefpo.types — core data types."""

from prefpo.types import (
    DiscriminatorResult,
    IterationRecord,
    ModelOutput,
    Prompt,
    PromptRole,
    Sample,
)
def test_prompt_creation():
    p = Prompt(value="test", role=PromptRole.USER)
    assert p.value == "test"
    assert p.role == PromptRole.USER
    assert p.name == ""
    assert p.metadata == {}
def test_prompt_with_metadata():
    p = Prompt(value="x", role=PromptRole.SYSTEM, name="p1", metadata={"k": "v"})
    assert p.name == "p1"
    assert p.metadata["k"] == "v"
def test_prompt_empty_value():
    p = Prompt(value="", role=PromptRole.USER)
    assert p.value == ""
def test_prompt_role_enum():
    assert PromptRole.USER.value == "user"
    assert PromptRole.SYSTEM.value == "system"
    assert PromptRole("user") == PromptRole.USER
def test_sample_creation():
    s = Sample(index=0, question="What?", target="42")
    assert s.index == 0
    assert s.question == "What?"
    assert s.target == "42"
def test_sample_defaults():
    s = Sample(index=5, question="Q")
    assert s.target == ""
    assert s.metadata == {}
def test_model_output():
    mo = ModelOutput(sample_index=3, prompt_sent="prompt", response="answer")
    assert mo.sample_index == 3
    assert mo.response == "answer"
def test_discriminator_result():
    dr = DiscriminatorResult(preferred=1, feedback="good")
    assert dr.preferred == 1
    assert dr.feedback == "good"
def test_discriminator_result_preferred_2():
    dr = DiscriminatorResult(preferred=2, feedback="ok")
    assert dr.preferred == 2
    assert dr.feedback == "ok"
def test_iteration_record():
    pa = Prompt(value="a", role=PromptRole.USER, name="a")
    pb = Prompt(value="b", role=PromptRole.USER, name="b")
    imp = Prompt(value="c", role=PromptRole.USER, name="improved_0")
    rec = IterationRecord(
        iteration=0,
        prompt_a=pa,
        prompt_b=pb,
        prompt_a_score=0.5,
        prompt_b_score=0.6,
        preferred=2,
        feedback="b is better",
        improved_prompt=imp,
        improved_score=0.7,
        discriminator_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        optimizer_usage={"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
    )
    assert rec.iteration == 0
    assert rec.preferred == 2
    assert rec.improved_score == 0.7
# --- Metadata validation edge cases ---

import pytest
def test_prompt_invalid_metadata_type():
    """List value in metadata raises TypeError."""
    with pytest.raises(TypeError, match="Metadata value"):
        Prompt(value="x", role=PromptRole.USER, metadata={"k": [1, 2, 3]})
def test_prompt_invalid_metadata_nested():
    """Dict value in metadata raises TypeError."""
    with pytest.raises(TypeError, match="Metadata value"):
        Prompt(value="x", role=PromptRole.USER, metadata={"k": {"nested": True}})
def test_sample_invalid_metadata():
    """Invalid metadata on Sample raises TypeError."""
    with pytest.raises(TypeError, match="Metadata value"):
        Sample(index=0, question="Q", metadata={"k": [1]})
def test_sample_frozen():
    """Assigning to Sample field raises FrozenInstanceError."""
    s = Sample(index=0, question="Q")
    with pytest.raises(AttributeError):
        s.index = 99
def test_model_output_frozen():
    """Assigning to ModelOutput field raises FrozenInstanceError."""
    mo = ModelOutput(sample_index=0, prompt_sent="p", response="r")
    with pytest.raises(AttributeError):
        mo.response = "new"
