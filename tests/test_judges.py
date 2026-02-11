"""Tests for prefpo.judges — prompt hack detection and hygiene scoring.

These tests make real API calls (no mocks). Requires OPENAI_API_KEY to be set.
Run with: pytest tests/test_judges.py -v
"""

import asyncio

import pytest
from prefpo.judges import judge_prompt_hack, judge_prompt_hygiene
pytestmark = pytest.mark.live

# --- Hack Detection ---
@pytest.mark.asyncio
async def test_hack_pass_simple():
    """A prompt that faithfully represents criteria should pass."""
    result = await judge_prompt_hack(
        prompt="Write a short poem about the ocean. Keep it under 100 words.",
        criteria=["Response must have less than 100 words"],
    )
    assert "reasoning" in result
    assert result["grade"] in ("pass", "fail")
    assert result["grade"] == "pass"
@pytest.mark.asyncio
async def test_hack_fail_over_restriction():
    """A prompt that tightens criteria beyond what's required should fail."""
    result = await judge_prompt_hack(
        prompt=(
            "Write a short poem about the ocean. Keep it under 50 words. "
            "Do not exceed 50 words under any circumstances."
        ),
        criteria=["Response must have less than 100 words"],
    )
    assert result["grade"] in ("pass", "fail")
    assert result["grade"] == "fail"
@pytest.mark.asyncio
async def test_hack_returns_correct_shape():
    """Return dict has exactly 'reasoning' and 'grade' keys."""
    result = await judge_prompt_hack(
        prompt="Write a haiku.",
        criteria=["Response must be a haiku"],
    )
    assert set(result.keys()) == {"reasoning", "grade"}
    assert isinstance(result["reasoning"], str)
    assert result["grade"] in ("pass", "fail")
@pytest.mark.asyncio
async def test_hack_grade_is_lowercase():
    """Grade should always be lowercase."""
    result = await judge_prompt_hack(
        prompt="Tell me a joke.",
        criteria=["Response should be funny"],
    )
    assert result["grade"] == result["grade"].lower()
# --- Hygiene Scoring ---
@pytest.mark.asyncio
async def test_hygiene_clean_prompt():
    """A clean, simple prompt should score well."""
    result = await judge_prompt_hygiene(
        prompt="Write a short blog post about a trip to Japan. Keep it under 300 words.",
    )
    assert "overall_reasoning" in result
    assert all(
        key in result
        for key in [
            "readability_reasoning",
            "readability_score",
            "spec_quality_reasoning",
            "spec_quality_score",
            "maintainability_reasoning",
            "maintainability_score",
        ]
    )
    # A clean prompt should get at least 1 on each axis
    assert result["readability_score"] >= 1
    assert result["spec_quality_score"] >= 1
    assert result["maintainability_score"] >= 1
@pytest.mark.asyncio
async def test_hygiene_messy_prompt():
    """A messy, over-specified prompt should score poorly."""
    result = await judge_prompt_hygiene(
        prompt=(
            "Write a blog post about Japan. Must be under 300 words. Do NOT exceed 300 words. "
            "Make sure to count your words. Target 250 words. Hard cap at 280. "
            "Do not use markdown. Do not use bullet points. Do not use headers. "
            "Before submitting, verify your word count is under 300. "
            "Re-check that no markdown is present. Output only plain text."
        ),
    )
    # A messy prompt should score 0 or 1 on spec quality
    assert result["spec_quality_score"] <= 1
@pytest.mark.asyncio
async def test_hygiene_returns_correct_shape():
    """Return dict has exactly 7 expected keys."""
    result = await judge_prompt_hygiene(prompt="Write a haiku about spring.")
    expected_keys = {
        "overall_reasoning",
        "readability_reasoning",
        "readability_score",
        "spec_quality_reasoning",
        "spec_quality_score",
        "maintainability_reasoning",
        "maintainability_score",
    }
    assert set(result.keys()) == expected_keys
@pytest.mark.asyncio
async def test_hygiene_scores_are_integers_0_to_2():
    """All scores should be integers in the range [0, 2]."""
    result = await judge_prompt_hygiene(prompt="Tell me a joke about programming.")
    for key in ["readability_score", "spec_quality_score", "maintainability_score"]:
        assert isinstance(result[key], int)
        assert 0 <= result[key] <= 2
@pytest.mark.asyncio
async def test_hygiene_with_context():
    """Context parameter should be accepted and influence evaluation."""
    result = await judge_prompt_hygiene(
        prompt=(
            "Write a blog post about clean energy. "
            "You must include the keywords 'solar' and 'wind'. "
            "Keep it under 200 words."
        ),
        context="Response must be under 200 words. Response must include 'solar' and 'wind'.",
    )
    assert "overall_reasoning" in result
    assert result["readability_score"] >= 0
@pytest.mark.asyncio
async def test_hygiene_without_context():
    """Works correctly when context is None (default)."""
    result = await judge_prompt_hygiene(
        prompt="Write a haiku about the moon.",
        context=None,
    )
    assert "overall_reasoning" in result
    assert result["readability_score"] >= 0
