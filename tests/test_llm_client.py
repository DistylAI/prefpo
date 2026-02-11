"""Tests for prefpo.llm.client — _convert_messages, _extract_usage, call_llm retries."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prefpo.llm.client import (
    LLMResponse,
    _convert_messages,
    _extract_usage,
    call_llm,
    call_llm_json,
)
def _make_litellm_response(text: str, response_id: str = "resp_test"):
    """Create a mock litellm response with the right structure."""
    text_block = MagicMock()
    text_block.type = "output_text"
    text_block.text = text

    output_item = MagicMock()
    output_item.content = [text_block]

    response = MagicMock()
    response.output = [output_item]
    response.id = response_id
    response.usage = MagicMock()
    response.usage.model_dump.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    return response
# --- _convert_messages ---
def test_convert_messages_non_reasoning():
    """For non-reasoning models, system stays system."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    result = _convert_messages(msgs, is_reasoning=False)
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
def test_convert_messages_reasoning():
    """For reasoning models, system becomes developer."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    result = _convert_messages(msgs, is_reasoning=True)
    assert result[0]["role"] == "developer"
    assert result[1]["role"] == "user"
def test_convert_messages_empty():
    """Empty list returns empty list."""
    assert _convert_messages([], is_reasoning=False) == []
    assert _convert_messages([], is_reasoning=True) == []
def test_convert_messages_content_format():
    """Output has input_text format."""
    msgs = [{"role": "user", "content": "hello"}]
    result = _convert_messages(msgs, is_reasoning=False)
    assert result[0]["content"] == [{"type": "input_text", "text": "hello"}]
# --- _extract_usage ---
def test_extract_usage_normal():
    """Extracts input/output/total tokens."""
    mock_resp = MagicMock()
    mock_resp.usage.model_dump.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    result = _extract_usage(mock_resp)
    assert result == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
def test_extract_usage_none():
    """response.usage=None returns zeros."""
    mock_resp = MagicMock()
    mock_resp.usage = None
    result = _extract_usage(mock_resp)
    assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
def test_extract_usage_missing_fields():
    """Partial usage dict returns 0 for missing fields."""
    mock_resp = MagicMock()
    mock_resp.usage.model_dump.return_value = {"input_tokens": 42}
    result = _extract_usage(mock_resp)
    assert result["input_tokens"] == 42
    assert result["output_tokens"] == 0
    assert result["total_tokens"] == 42  # computed as input + output
# --- call_llm retries ---
@pytest.mark.asyncio
async def test_call_llm_retries_on_rate_limit():
    """Retries on RateLimitError then succeeds."""
    from litellm.exceptions import RateLimitError

    mock_response = _make_litellm_response("ok")
    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError(
                message="rate limit",
                model="gpt-4o",
                llm_provider="openai",
            )
        return mock_response

    with patch("prefpo.llm.client.litellm.aresponses", new_callable=AsyncMock, side_effect=side_effect):
        with patch("prefpo.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await call_llm(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                max_retries=3,
            )
    assert result.output_text == "ok"
    assert call_count == 2
@pytest.mark.asyncio
async def test_call_llm_retries_on_server_error():
    """Retries on InternalServerError then succeeds."""
    from litellm.exceptions import InternalServerError

    mock_response = _make_litellm_response("ok")
    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise InternalServerError(
                message="server error",
                model="gpt-4o",
                llm_provider="openai",
            )
        return mock_response

    with patch("prefpo.llm.client.litellm.aresponses", new_callable=AsyncMock, side_effect=side_effect):
        with patch("prefpo.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await call_llm(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                max_retries=3,
            )
    assert result.output_text == "ok"
@pytest.mark.asyncio
async def test_call_llm_raises_after_max_retries():
    """Raises after exhausting retries."""
    from litellm.exceptions import RateLimitError

    mock_aresponses = AsyncMock(
        side_effect=RateLimitError(
            message="rate limit",
            model="gpt-4o",
            llm_provider="openai",
        )
    )

    with patch("prefpo.llm.client.litellm.aresponses", mock_aresponses):
        with patch("prefpo.llm.client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RateLimitError):
                await call_llm(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    max_retries=2,
                )
    assert mock_aresponses.call_count == 2
# --- call_llm_json ---
@pytest.mark.asyncio
async def test_call_llm_json_retries_parse_failure():
    """Retries on JSONDecodeError then succeeds."""
    bad_resp = LLMResponse(output_text="not json", id="resp_bad", usage=None)
    good_resp = LLMResponse(output_text='{"ok": true}', id="resp_good", usage=None)

    with patch("prefpo.llm.client.call_llm", new_callable=AsyncMock, side_effect=[bad_resp, good_resp]):
        parsed, resp = await call_llm_json(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            parse_retries=1,
        )
    assert parsed == {"ok": True}
    assert resp.output_text == '{"ok": true}'
@pytest.mark.asyncio
async def test_call_llm_json_returns_tuple():
    """Returns (parsed_dict, LLMResponse) tuple."""
    mock_resp = LLMResponse(output_text='{"key": "value"}', id="resp_test", usage=None)

    with patch("prefpo.llm.client.call_llm", new_callable=AsyncMock, return_value=mock_resp):
        parsed, resp = await call_llm_json(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert isinstance(parsed, dict)
    assert parsed["key"] == "value"
    assert resp.output_text == '{"key": "value"}'
    assert resp.id == "resp_test"
