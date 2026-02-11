"""Tests for run_iteration() with mocked LLM calls."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prefpo.config import DiscriminatorConfig, ModelConfig, OptimizerConfig
from prefpo.optimize import run_iteration
from prefpo.types import ModelOutput, Prompt, PromptRole, Sample
def _make_mock_response(output_text: str, response_id: str = "resp_test"):
    """Create a mock LLM response object."""
    mock = MagicMock()
    mock.output_text = output_text
    mock.id = response_id
    mock.usage = MagicMock()
    mock.usage.model_dump.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    return mock
@pytest.fixture
def semaphore():
    return asyncio.Semaphore(10)
@pytest.fixture
def basic_prompts():
    return (
        Prompt(value="Prompt A", role=PromptRole.USER, name="seed_0"),
        Prompt(value="Prompt B", role=PromptRole.USER, name="seed_1"),
    )
@pytest.fixture
def train_samples():
    return [
        Sample(index=0, question="What is 2+2?", target="4"),
        Sample(index=1, question="What is 3+3?", target="6"),
    ]
@pytest.mark.asyncio
async def test_run_iteration_instruction_mode(basic_prompts, train_samples, semaphore):
    prompt_a, prompt_b = basic_prompts

    disc_json = json.dumps({"preferred": 1, "feedback": "A is better at math"})
    opt_json = json.dumps({"prompt": "Improved instruction for math"})

    disc_resp = _make_mock_response(disc_json, "resp_disc_123")
    opt_resp = _make_mock_response(opt_json, "resp_opt_456")

    mock_outputs = [
        ModelOutput(sample_index=0, prompt_sent="p", response="ANSWER: 4"),
        ModelOutput(sample_index=1, prompt_sent="p", response="ANSWER: 6"),
    ]

    with patch("prefpo.optimize.generate_outputs", new_callable=AsyncMock, return_value=mock_outputs), \
         patch("prefpo.optimize.call_discriminator_with_messages", new_callable=AsyncMock, return_value=(
             json.loads(disc_json),
             [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": disc_json}],
             disc_resp,
         )), \
         patch("prefpo.optimize.call_optimizer_with_messages", new_callable=AsyncMock, return_value=(
             json.loads(opt_json), opt_resp,
         )):

        result = await run_iteration(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            mode="instruction",
            train_samples=train_samples,
            grader=MagicMock(),
            task_model=ModelConfig(name="openai/gpt-4o"),
            disc_config=DiscriminatorConfig(),
            opt_config=OptimizerConfig(),
            iteration_index=0,
            semaphore=semaphore,
        )

    assert result.preferred == 1
    assert result.feedback == "A is better at math"
    assert result.improved_prompt.value == "Improved instruction for math"
    assert result.improved_prompt.name == "improved_0"
    assert result.discriminator_usage["input_tokens"] == 100
    assert result.optimizer_usage["total_tokens"] == 150
@pytest.mark.asyncio
async def test_run_iteration_standalone_mode(semaphore):
    prompt_a = Prompt(value="Write a poem", role=PromptRole.USER, name="seed_0")
    prompt_b = Prompt(value="Write a nice poem", role=PromptRole.USER, name="seed_1")

    disc_json = json.dumps({"preferred": 2, "feedback": "B is more creative"})
    opt_json = json.dumps({"prompt": "Write a beautiful poem"})

    disc_resp = _make_mock_response(disc_json, "resp_disc")
    opt_resp = _make_mock_response(opt_json, "resp_opt")

    mock_outputs = [ModelOutput(sample_index=-1, prompt_sent="p", response="A nice poem")]

    mock_grader = MagicMock()
    mock_grader.check_output.return_value = {"passed": True}

    with patch("prefpo.optimize.generate_standalone", new_callable=AsyncMock, return_value=mock_outputs), \
         patch("prefpo.optimize.call_discriminator_with_messages", new_callable=AsyncMock, return_value=(
             json.loads(disc_json),
             [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": disc_json}],
             disc_resp,
         )), \
         patch("prefpo.optimize.call_optimizer_with_messages", new_callable=AsyncMock, return_value=(
             json.loads(opt_json), opt_resp,
         )):

        result = await run_iteration(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            mode="standalone",
            train_samples=None,
            grader=mock_grader,
            task_model=ModelConfig(name="openai/gpt-4o"),
            disc_config=DiscriminatorConfig(show_target=True),
            opt_config=OptimizerConfig(),
            iteration_index=3,
            semaphore=semaphore,
        )

    assert result.preferred == 2
    assert result.improved_prompt.name == "improved_3"
    assert result.improved_prompt.value == "Write a beautiful poem"
@pytest.mark.asyncio
async def test_run_iteration_empty_optimizer_output(basic_prompts, train_samples, semaphore):
    """When optimizer returns empty string, should fall back to non-preferred prompt."""
    prompt_a, prompt_b = basic_prompts

    disc_json = json.dumps({"preferred": 1, "feedback": "A is better"})
    opt_json = json.dumps({"prompt": ""})

    disc_resp = _make_mock_response(disc_json, "resp_disc")
    opt_resp = _make_mock_response(opt_json, "resp_opt")

    mock_outputs = [
        ModelOutput(sample_index=0, prompt_sent="p", response="resp"),
        ModelOutput(sample_index=1, prompt_sent="p", response="resp"),
    ]

    with patch("prefpo.optimize.generate_outputs", new_callable=AsyncMock, return_value=mock_outputs), \
         patch("prefpo.optimize.call_discriminator_with_messages", new_callable=AsyncMock, return_value=(
             json.loads(disc_json),
             [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": disc_json}],
             disc_resp,
         )), \
         patch("prefpo.optimize.call_optimizer_with_messages", new_callable=AsyncMock, return_value=(
             json.loads(opt_json), opt_resp,
         )):

        result = await run_iteration(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            mode="instruction",
            train_samples=train_samples,
            grader=MagicMock(),
            task_model=ModelConfig(name="openai/gpt-4o"),
            disc_config=DiscriminatorConfig(),
            opt_config=OptimizerConfig(),
            iteration_index=0,
            semaphore=semaphore,
        )

    # Empty optimizer output should fall back to non-preferred prompt (prompt_b since preferred=1)
    assert result.improved_prompt.value == prompt_b.value
# --- Additional edge case tests with real API ---

from prefpo.data.bbh import load_bbh
from prefpo.grading import get_bbh_grader
@pytest.mark.asyncio
@pytest.mark.live
async def test_run_iteration_preferred_2():
    """Real API: when discriminator picks version 2, optimizer improves version 1."""
    train, _, _ = load_bbh("disambiguation_qa", train_size=2, val_size=1, seed=42)
    grader = get_bbh_grader("disambiguation_qa")

    prompt_a = Prompt(value="Answer the question. End with ANSWER: <letter>.", role=PromptRole.USER, name="seed_0")
    prompt_b = Prompt(value="Think step by step. End with ANSWER: <letter>.", role=PromptRole.USER, name="seed_1")

    result = await run_iteration(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        mode="instruction",
        train_samples=train,
        grader=grader,
        task_model=ModelConfig(name="openai/gpt-4o"),
        disc_config=DiscriminatorConfig(),
        opt_config=OptimizerConfig(),
        iteration_index=0,
        semaphore=asyncio.Semaphore(10),
    )

    assert result.preferred in (1, 2)
    assert len(result.feedback) > 0
    assert len(result.improved_prompt.value) > 0
    assert result.improved_prompt.name == "improved_0"
@pytest.mark.asyncio
@pytest.mark.live
async def test_run_iteration_token_usage():
    """Real API: discriminator_usage and optimizer_usage have expected keys."""
    train, _, _ = load_bbh("disambiguation_qa", train_size=2, val_size=1, seed=42)
    grader = get_bbh_grader("disambiguation_qa")

    prompt_a = Prompt(value="Answer correctly. End with ANSWER: <letter>.", role=PromptRole.USER, name="a")
    prompt_b = Prompt(value="Be careful. End with ANSWER: <letter>.", role=PromptRole.USER, name="b")

    result = await run_iteration(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        mode="instruction",
        train_samples=train,
        grader=grader,
        task_model=ModelConfig(name="openai/gpt-4o"),
        disc_config=DiscriminatorConfig(),
        opt_config=OptimizerConfig(),
        iteration_index=0,
        semaphore=asyncio.Semaphore(10),
    )

    for usage_dict in (result.discriminator_usage, result.optimizer_usage):
        assert "input_tokens" in usage_dict
        assert "output_tokens" in usage_dict
        assert "total_tokens" in usage_dict
        assert usage_dict["input_tokens"] > 0
        assert usage_dict["output_tokens"] > 0
