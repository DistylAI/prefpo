"""Cross-provider tests — verify litellm works with OpenAI and Anthropic.

These tests make real API calls and are NOT mocked. They are excluded from
the default test run. To run them:

    pytest -m live tests/test_multi_provider.py -v
"""

import asyncio

import pytest

pytestmark = pytest.mark.live

from prefpo.config import DiscriminatorConfig, ModelConfig, OptimizerConfig
from prefpo.data.bbh import load_bbh
from prefpo.generate import generate_outputs, generate_standalone
from prefpo.grading import get_bbh_grader
from prefpo.llm.client import (
    call_discriminator_with_messages,
    call_llm,
    call_llm_json,
    call_optimizer_with_messages,
)
from prefpo.optimize import run_iteration
from prefpo.prompts.discriminator import DISCRIMINATOR_SCHEMA
from prefpo.prompts.optimizer import OPTIMIZER_SCHEMA
from prefpo.types import Prompt, PromptRole, Sample
# --- call_llm across providers ---
@pytest.mark.asyncio
async def test_call_llm_anthropic():
    """call_llm works with Anthropic Claude."""
    response = await call_llm(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    )
    assert len(response.output_text) > 0
    assert "4" in response.output_text
@pytest.mark.asyncio
async def test_call_llm_openai():
    """call_llm works with OpenAI."""
    response = await call_llm(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    )
    assert len(response.output_text) > 0
    assert "4" in response.output_text
# --- call_llm_json across providers ---
@pytest.mark.asyncio
async def test_call_llm_json_anthropic():
    """JSON structured output works with Anthropic."""
    parsed, resp = await call_llm_json(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Return JSON: {\"answer\": 4}"}],
        json_schema={
            "type": "json_schema",
            "name": "answer_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "number"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    )
    assert parsed["answer"] == 4
    assert len(resp.output_text) > 0
@pytest.mark.asyncio
async def test_call_llm_json_openai():
    """JSON structured output works with OpenAI."""
    parsed, resp = await call_llm_json(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Return JSON: {\"answer\": 4}"}],
        json_schema={
            "type": "json_schema",
            "name": "answer_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "number"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    )
    assert parsed["answer"] == 4
# --- generate functions with Anthropic ---
@pytest.mark.asyncio
async def test_generate_outputs_anthropic():
    """generate_outputs works with Anthropic model."""
    p = Prompt(value="You are helpful.", role=PromptRole.SYSTEM)
    s = Sample(index=0, question="What is 2+2? Answer with just the number.")
    outputs = await generate_outputs(
        p, [s],
        ModelConfig(name="anthropic/claude-haiku-4-5-20251001"),
        asyncio.Semaphore(10),
    )
    assert len(outputs) == 1
    assert outputs[0].sample_index == 0
    assert "4" in outputs[0].response
@pytest.mark.asyncio
async def test_generate_standalone_anthropic():
    """generate_standalone works with Anthropic model."""
    p = Prompt(value="Say hello in exactly 3 words.", role=PromptRole.USER)
    outputs = await generate_standalone(
        p,
        ModelConfig(name="anthropic/claude-haiku-4-5-20251001"),
        asyncio.Semaphore(10),
        n=1,
    )
    assert len(outputs) == 1
    assert outputs[0].sample_index == -1
    assert len(outputs[0].response) > 0
# --- System/developer role with Anthropic ---
@pytest.mark.asyncio
async def test_call_llm_anthropic_system_role():
    """Anthropic handles system messages correctly."""
    response = await call_llm(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[
            {"role": "system", "content": "You always respond with exactly one word."},
            {"role": "user", "content": "What color is the sky?"},
        ],
    )
    assert len(response.output_text) > 0
    # Should be a short response given the system prompt
    assert len(response.output_text.split()) <= 5
# --- Anthropic as discriminator ---
@pytest.mark.asyncio
async def test_discriminator_anthropic():
    """call_discriminator_with_messages works with Anthropic model."""
    messages = [
        {"role": "system", "content": "You are a judge. Compare two versions and pick the better one."},
        {"role": "user", "content": (
            'Version 1 output: "4"\nVersion 2 output: "four"\n'
            "Which version is better? Return JSON with preferred (1 or 2) and feedback."
        )},
    ]
    parsed, msgs_out, resp = await call_discriminator_with_messages(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=messages,
        json_schema=DISCRIMINATOR_SCHEMA,
    )
    assert parsed["preferred"] in (1, 2)
    assert len(parsed["feedback"]) > 0
    # Messages should include the assistant response appended
    assert msgs_out[-1]["role"] == "assistant"
    assert len(resp.output_text) > 0
# --- Anthropic as optimizer ---
@pytest.mark.asyncio
async def test_optimizer_anthropic():
    """call_optimizer_with_messages works with Anthropic model."""
    # Simulate disc_messages (system + user + assistant)
    disc_messages = [
        {"role": "system", "content": "You are a judge."},
        {"role": "user", "content": "Version 1: 'Answer briefly.' Version 2: 'Answer in detail.'"},
        {"role": "assistant", "content": '{"preferred": 1, "feedback": "Version 1 is more concise."}'},
    ]
    parsed, resp = await call_optimizer_with_messages(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=disc_messages,
        optimizer_prompt=(
            "Improve the non-preferred prompt based on the feedback. "
            'Return JSON: {"prompt": "<improved prompt>"}'
        ),
        json_schema=OPTIMIZER_SCHEMA,
    )
    assert "prompt" in parsed
    assert len(parsed["prompt"]) > 0
    assert len(resp.output_text) > 0
# --- Full run_iteration with Anthropic for all three roles ---
@pytest.mark.asyncio
async def test_run_iteration_all_anthropic():
    """Full iteration with Anthropic for task model, discriminator, and optimizer."""
    train, _, _ = load_bbh("disambiguation_qa", train_size=2, val_size=1, seed=42)
    grader = get_bbh_grader("disambiguation_qa")

    anthropic_model = ModelConfig(name="anthropic/claude-haiku-4-5-20251001")

    prompt_a = Prompt(
        value="Answer the question. End with ANSWER: <letter>.",
        role=PromptRole.USER, name="seed_0",
    )
    prompt_b = Prompt(
        value="Think step by step. End with ANSWER: <letter>.",
        role=PromptRole.USER, name="seed_1",
    )

    result = await run_iteration(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        mode="instruction",
        train_samples=train,
        grader=grader,
        task_model=anthropic_model,
        disc_config=DiscriminatorConfig(model=anthropic_model, show_target=True),
        opt_config=OptimizerConfig(model=anthropic_model),
        iteration_index=0,
        semaphore=asyncio.Semaphore(10),
    )

    assert result.preferred in (1, 2)
    assert len(result.feedback) > 0
    assert len(result.improved_prompt.value) > 0
    assert result.improved_prompt.name == "improved_0"
    assert result.discriminator_usage["input_tokens"] > 0
    assert result.optimizer_usage["input_tokens"] > 0
# --- Reasoning model as task model ---
@pytest.mark.asyncio
async def test_call_llm_reasoning_model():
    """Reasoning model works via is_reasoning=True."""
    response = await call_llm(
        model="openai/o4-mini",
        messages=[{"role": "user", "content": "What is 17 * 23?"}],
        is_reasoning=True,
        reasoning_effort="low",
    )
    assert len(response.output_text) > 0
    assert "391" in response.output_text
@pytest.mark.asyncio
async def test_generate_outputs_reasoning_model():
    """generate_outputs works with a reasoning model as task model."""
    p = Prompt(
        value="Answer the question. End with ANSWER: <letter>.",
        role=PromptRole.USER,
    )
    s = Sample(index=0, question="Is the sky blue? (A) Yes (B) No")
    outputs = await generate_outputs(
        p, [s],
        ModelConfig(name="openai/o4-mini", is_reasoning=True, reasoning_effort="low"),
        asyncio.Semaphore(10),
    )
    assert len(outputs) == 1
    assert len(outputs[0].response) > 0
