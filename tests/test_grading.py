"""Tests for prefpo.grading — MC, binary, exact_match graders."""

from prefpo.grading import get_bbh_grader
from prefpo.grading.binary import BinaryGrader, _parse_answer_word
from prefpo.grading.exact_match import ExactMatchGrader, _match_end
from prefpo.grading.multiple_choice import MultipleChoiceGrader, _parse_answer_letter
# --- MC parsing ---
def test_mc_parse_strict():
    assert _parse_answer_letter("ANSWER: A") == "A"
    assert _parse_answer_letter("ANSWER: B") == "B"
    assert _parse_answer_letter("ANSWER: C") == "C"
def test_mc_parse_loose():
    assert _parse_answer_letter("The answer is ANSWER: A.") == "A"
    assert _parse_answer_letter("I think ANSWER: B") == "B"
def test_mc_parse_case_insensitive():
    assert _parse_answer_letter("answer: a") == "A"
def test_mc_parse_no_match():
    assert _parse_answer_letter("I think it's A") is None
    assert _parse_answer_letter("No answer here") is None
# --- Binary parsing ---
def test_binary_parse_yes_no():
    assert _parse_answer_word("ANSWER: Yes") == "yes"
    assert _parse_answer_word("ANSWER: No") == "no"
def test_binary_parse_true_false():
    assert _parse_answer_word("ANSWER: True") == "true"
    assert _parse_answer_word("ANSWER: False") == "false"
def test_binary_parse_valid_invalid():
    assert _parse_answer_word("ANSWER: valid") == "valid"
    assert _parse_answer_word("ANSWER: invalid") == "invalid"
def test_binary_parse_no_match():
    assert _parse_answer_word("I think yes") is None
# --- Exact match ---
def test_match_end_basic():
    assert _match_end("The answer is 42", "42") is True
    assert _match_end("The answer is 42.", "42") is True
def test_match_end_case_insensitive():
    assert _match_end("apple banana cherry", "Cherry") is True
def test_match_end_negative():
    assert _match_end("42 is the start", "42") is False
def test_match_end_punctuation():
    assert _match_end("The answer is: 42!", "42") is True
# --- check_output ---
def test_mc_check_output_pred():
    g = MultipleChoiceGrader()
    result = g.check_output("ANSWER: A", "prompt")
    assert result == {"pred": "A"}
def test_binary_check_output_pred():
    g = BinaryGrader()
    result = g.check_output("ANSWER: yes", "prompt")
    assert result == {"pred": "yes"}
def test_exact_match_check_output_none():
    g = ExactMatchGrader()
    result = g.check_output("The answer is 42", "prompt")
    assert result is None
# --- Factory ---
def test_get_bbh_grader_mc():
    assert isinstance(get_bbh_grader("disambiguation_qa"), MultipleChoiceGrader)
def test_get_bbh_grader_binary():
    assert isinstance(get_bbh_grader("navigate"), BinaryGrader)
def test_get_bbh_grader_exact():
    assert isinstance(get_bbh_grader("object_counting"), ExactMatchGrader)
def test_get_bbh_grader_unknown():
    import pytest
    with pytest.raises(ValueError):
        get_bbh_grader("not_a_task")
# --- Grader.grade() with real API calls ---

import asyncio
from unittest.mock import AsyncMock

import pytest
from prefpo.config import ModelConfig
from prefpo.data.bbh import load_bbh
from prefpo.types import Prompt, PromptRole, Sample
@pytest.mark.asyncio
@pytest.mark.live
async def test_mc_grade_real():
    """MC grader with real API call on disambiguation_qa samples."""
    grader = MultipleChoiceGrader()
    train, _, _ = load_bbh("disambiguation_qa", train_size=3, val_size=1, seed=42)
    result = await grader.grade(
        Prompt(value="Answer the following question. End with ANSWER: <letter>.", role=PromptRole.USER),
        train,
        ModelConfig(name="openai/gpt-4o"),
        asyncio.Semaphore(10),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.n == 3
    assert len(result.per_sample) == 3
    for ps in result.per_sample:
        assert "pred" in ps
        assert "target" in ps
        assert "correct" in ps
@pytest.mark.asyncio
@pytest.mark.live
async def test_mc_grade_empty_target_raises():
    """MC grader raises ValueError if sample.target is empty."""
    grader = MultipleChoiceGrader()
    samples = [Sample(index=0, question="Q", target="")]
    with pytest.raises(ValueError, match="empty target"):
        await grader.grade(
            Prompt(value="x", role=PromptRole.USER), samples,
            ModelConfig(name="openai/gpt-4o"), asyncio.Semaphore(10),
        )
@pytest.mark.asyncio
@pytest.mark.live
async def test_binary_grade_real():
    """Binary grader with real API call on navigate samples."""
    grader = BinaryGrader()
    train, _, _ = load_bbh("navigate", train_size=3, val_size=1, seed=42)
    result = await grader.grade(
        Prompt(value="Answer the following question. End with ANSWER: <yes/no>.", role=PromptRole.USER),
        train,
        ModelConfig(name="openai/gpt-4o"),
        asyncio.Semaphore(10),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.n == 3
    assert len(result.per_sample) == 3
@pytest.mark.asyncio
@pytest.mark.live
async def test_binary_grade_empty_target_raises():
    """Binary grader raises ValueError if sample.target is empty."""
    grader = BinaryGrader()
    samples = [Sample(index=0, question="Q", target="")]
    with pytest.raises(ValueError, match="empty target"):
        await grader.grade(
            Prompt(value="x", role=PromptRole.USER), samples,
            ModelConfig(name="openai/gpt-4o"), asyncio.Semaphore(10),
        )
@pytest.mark.asyncio
@pytest.mark.live
async def test_exact_match_grade_real():
    """ExactMatch grader with real API call on object_counting samples."""
    grader = ExactMatchGrader()
    train, _, _ = load_bbh("object_counting", train_size=3, val_size=1, seed=42)
    result = await grader.grade(
        Prompt(value="Count the objects. End your response with the number.", role=PromptRole.USER),
        train,
        ModelConfig(name="openai/gpt-4o"),
        asyncio.Semaphore(10),
    )
    assert 0.0 <= result.score <= 1.0
    assert result.n == 3
    assert result.stderr >= 0.0
@pytest.mark.asyncio
@pytest.mark.live
async def test_exact_match_grade_empty_target_raises():
    """ExactMatch grader raises ValueError if sample.target is empty."""
    grader = ExactMatchGrader()
    samples = [Sample(index=0, question="Q", target="")]
    with pytest.raises(ValueError, match="empty target"):
        await grader.grade(
            Prompt(value="x", role=PromptRole.USER), samples,
            ModelConfig(name="openai/gpt-4o"), asyncio.Semaphore(10),
        )
