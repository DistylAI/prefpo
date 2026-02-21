"""Tests for prefpo.generate — message formatting."""

import pytest
from prefpo.generate import format_instruction_messages, format_standalone_messages
from prefpo.types import Prompt, PromptRole, Sample
def test_instruction_user_role():
    p = Prompt(value="Be careful.", role=PromptRole.USER)
    s = Sample(index=0, question="What is 2+2?")
    msgs = format_instruction_messages(p, s)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "Be careful." in msgs[0]["content"]
    assert "What is 2+2?" in msgs[0]["content"]
def test_instruction_system_role():
    p = Prompt(value="You are helpful.", role=PromptRole.SYSTEM)
    s = Sample(index=0, question="What is 2+2?")
    msgs = format_instruction_messages(p, s)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "What is 2+2?"
def test_instruction_empty_prompt_user():
    p = Prompt(value="", role=PromptRole.USER)
    s = Sample(index=0, question="What is 2+2?")
    msgs = format_instruction_messages(p, s)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "What is 2+2?"
def test_instruction_empty_prompt_system():
    p = Prompt(value="", role=PromptRole.SYSTEM)
    s = Sample(index=0, question="What is 2+2?")
    msgs = format_instruction_messages(p, s)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is 2+2?"
def test_standalone_user_role():
    p = Prompt(value="Write a poem.", role=PromptRole.USER)
    msgs = format_standalone_messages(p)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Write a poem."
def test_standalone_system_role_rejected():
    p = Prompt(value="Write a poem.", role=PromptRole.SYSTEM)
    with pytest.raises(ValueError, match="Standalone mode requires"):
        format_standalone_messages(p)
# --- system_prompt tests ---

def test_instruction_user_role_with_system_prompt():
    """system_prompt adds a system message before the user message."""
    p = Prompt(value="Be careful.", role=PromptRole.USER)
    s = Sample(index=0, question="What is 2+2?")
    msgs = format_instruction_messages(p, s, system_prompt="You are a math tutor.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are a math tutor."
    assert msgs[1]["role"] == "user"
    assert "Be careful." in msgs[1]["content"]
    assert "What is 2+2?" in msgs[1]["content"]

def test_instruction_user_role_without_system_prompt_unchanged():
    """No system_prompt — behavior unchanged (single user message)."""
    p = Prompt(value="Be careful.", role=PromptRole.USER)
    s = Sample(index=0, question="What is 2+2?")
    msgs = format_instruction_messages(p, s, system_prompt=None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"

def test_standalone_with_system_prompt():
    """system_prompt adds a system message before the user prompt."""
    p = Prompt(value="Write a poem.", role=PromptRole.USER)
    msgs = format_standalone_messages(p, system_prompt="You are a poet.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are a poet."
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Write a poem."

def test_standalone_without_system_prompt_unchanged():
    """No system_prompt — behavior unchanged (single user message)."""
    p = Prompt(value="Write a poem.", role=PromptRole.USER)
    msgs = format_standalone_messages(p, system_prompt=None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"

def test_format_prompt_sent_includes_system_prompt():
    """prompt_sent captures system_prompt content."""
    p = Prompt(value="Be careful.", role=PromptRole.USER)
    s = Sample(index=0, question="What is 2+2?")
    msgs = format_instruction_messages(p, s, system_prompt="You are a math tutor.")
    sent = _format_prompt_sent(msgs)
    assert "[system] You are a math tutor." in sent
    assert "[user]" in sent

# --- _format_prompt_sent tests ---

from prefpo.generate import _format_prompt_sent
def test_format_prompt_sent_single_message():
    msgs = [{"role": "user", "content": "hello"}]
    assert _format_prompt_sent(msgs) == "[user] hello"
def test_format_prompt_sent_multiple_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert _format_prompt_sent(msgs) == "[system] sys\n\n[user] usr"
def test_format_prompt_sent_empty():
    assert _format_prompt_sent([]) == ""
# --- generate_outputs / generate_standalone with real API ---

import asyncio

from prefpo.config import ModelConfig
from prefpo.generate import generate_outputs, generate_standalone
@pytest.mark.asyncio
@pytest.mark.live
async def test_generate_outputs_real():
    """generate_outputs makes real API call and returns ModelOutput list."""
    p = Prompt(value="You are helpful.", role=PromptRole.SYSTEM)
    s = Sample(index=0, question="What is 2+2?")
    outputs = await generate_outputs(
        p, [s],
        ModelConfig(name="openai/gpt-4o"),
        asyncio.Semaphore(10),
    )
    assert len(outputs) == 1
    assert outputs[0].sample_index == 0
    assert "[system] You are helpful." in outputs[0].prompt_sent
    assert "[user] What is 2+2?" in outputs[0].prompt_sent
    assert len(outputs[0].response) > 0
@pytest.mark.asyncio
@pytest.mark.live
async def test_generate_standalone_real():
    """generate_standalone makes real API call, sample_index=-1."""
    p = Prompt(value="Write a haiku about the ocean.", role=PromptRole.USER)
    outputs = await generate_standalone(
        p,
        ModelConfig(name="openai/gpt-4o"),
        asyncio.Semaphore(10),
        n=1,
    )
    assert len(outputs) == 1
    assert outputs[0].sample_index == -1
    assert len(outputs[0].response) > 0
@pytest.mark.asyncio
@pytest.mark.live
async def test_generate_standalone_n_multiple():
    """n=3 produces 3 outputs."""
    p = Prompt(value="Say hello.", role=PromptRole.USER)
    outputs = await generate_standalone(
        p,
        ModelConfig(name="openai/gpt-4o"),
        asyncio.Semaphore(10),
        n=3,
    )
    assert len(outputs) == 3
    for o in outputs:
        assert o.sample_index == -1
        assert len(o.response) > 0

# --- system_prompt live API tests ---

@pytest.mark.asyncio
@pytest.mark.live
async def test_generate_outputs_with_system_prompt_keyword():
    """Instruction mode: system_prompt keyword appears in model output."""
    p = Prompt(value="Answer the question in one sentence.", role=PromptRole.USER)
    s = Sample(index=0, question="What color is the sky?")
    outputs = await generate_outputs(
        p, [s],
        ModelConfig(
            name="openai/gpt-4o",
            system_prompt="IMPORTANT: End every sentence with the word XYLOPHONE.",
        ),
        asyncio.Semaphore(10),
    )
    assert len(outputs) == 1
    assert "[system] IMPORTANT: End every sentence" in outputs[0].prompt_sent
    assert "XYLOPHONE" in outputs[0].response.upper()

@pytest.mark.asyncio
@pytest.mark.live
async def test_generate_standalone_with_system_prompt_keyword():
    """Standalone mode: system_prompt keyword appears in model output."""
    p = Prompt(value="Write one sentence about dogs.", role=PromptRole.USER)
    outputs = await generate_standalone(
        p,
        ModelConfig(
            name="openai/gpt-4o",
            system_prompt="IMPORTANT: End every sentence with the word XYLOPHONE.",
        ),
        asyncio.Semaphore(10),
        n=1,
    )
    assert len(outputs) == 1
    assert "[system] IMPORTANT: End every sentence" in outputs[0].prompt_sent
    assert "XYLOPHONE" in outputs[0].response.upper()
