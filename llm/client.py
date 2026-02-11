"""LLM API wrapper for PrefPO, backed by litellm.

Supports any litellm-compatible provider (OpenAI, Anthropic, DeepSeek, Gemini,
etc.) via the Responses API. Model names use litellm's provider/model format,
e.g. "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929".
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Uniform response wrapper over litellm's ResponsesAPIResponse.

    Provides the same interface callers expect (output_text, id, usage)
    regardless of provider.
    """

    output_text: str
    id: str
    usage: Any


def _get_output_text(response: Any) -> str:
    """Extract output text from a litellm ResponsesAPIResponse.

    litellm does NOT have .output_text — we walk response.output[i].content[j]
    looking for type=="output_text". Reasoning model responses have items with
    content=None (reasoning summary items) which we skip.
    """
    for item in response.output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for block in content:
            if getattr(block, "type", None) == "output_text":
                return getattr(block, "text", "") or ""
    return ""


def _convert_messages(
    messages: list[dict[str, str]], is_reasoning: bool
) -> list[dict[str, Any]]:
    """Convert simple messages to Responses API input format.

    Callers pass: [{"role": "system", "content": "..."}]
    We convert to: [{"role": "system"|"developer", "content": [{"type": "input_text", "text": "..."}]}]

    Assistant messages use "output_text" instead of "input_text" per the
    Responses API spec.
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        if role == "system" and is_reasoning:
            role = "developer"
        content_type = "output_text" if role == "assistant" else "input_text"
        converted.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": msg["content"]}],
            }
        )
    return converted


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from an LLM response.

    Works with both LLMResponse (access .usage) and raw litellm responses.
    """
    usage = response.usage
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    input_t = usage_dict.get("input_tokens", 0) or 0
    output_t = usage_dict.get("output_tokens", 0) or 0
    total = usage_dict.get("total_tokens", 0) or (input_t + output_t)
    return {"input_tokens": input_t, "output_tokens": output_t, "total_tokens": total}


async def call_llm(
    *,
    model: str,
    messages: list[dict[str, str]],
    is_reasoning: bool = False,
    reasoning_effort: str = "medium",
    temperature: float = 0.0,
    json_schema: dict | None = None,
    max_retries: int = 3,
) -> LLMResponse:
    """Call an LLM via litellm's Responses API with retries.

    Args:
        model: litellm model string (e.g. "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929").
        messages: Simple format: [{"role": "system"|"user", "content": "..."}].
        is_reasoning: Whether this is a reasoning model (affects role mapping and params).
        reasoning_effort: Reasoning effort level for reasoning models.
        temperature: Temperature for non-reasoning models (ignored for reasoning).
        json_schema: JSON schema dict for structured output.
        max_retries: Max retry attempts on transient errors.

    Returns:
        LLMResponse with output_text, id, and usage fields.
    """
    converted_input = _convert_messages(messages, is_reasoning)

    if json_schema is not None:
        text_format = {"format": json_schema}
    else:
        text_format = {"format": {"type": "text"}}

    kwargs: dict[str, Any] = {
        "model": model,
        "input": converted_input,
        "text": text_format,
    }

    if is_reasoning:
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
    else:
        kwargs["temperature"] = temperature

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await litellm.aresponses(**kwargs)
            return LLMResponse(
                output_text=_get_output_text(response),
                id=response.id,
                usage=response.usage,
            )
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            last_error = e
            wait = 2 ** (attempt + 1)
            logger.warning(
                "call_llm attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1,
                max_retries,
                e,
                wait,
            )
            await asyncio.sleep(wait)

    raise last_error  # type: ignore[misc]


async def call_llm_json(
    *,
    parse_retries: int = 1,
    **kwargs: Any,
) -> tuple[dict[str, Any], LLMResponse]:
    """Call call_llm() and parse the JSON output, retrying on parse failure.

    Args:
        parse_retries: Number of extra attempts if JSON parsing fails.
        **kwargs: Forwarded to call_llm().

    Returns:
        (parsed_dict, LLMResponse) tuple.

    Raises:
        json.JSONDecodeError: If parsing fails after all retries.
    """
    last_error: json.JSONDecodeError | None = None
    for attempt in range(1 + parse_retries):
        response = await call_llm(**kwargs)
        try:
            parsed = json.loads(response.output_text)
            return parsed, response
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                "JSON parse failed (attempt %d/%d): %s | raw text: %.200s",
                attempt + 1,
                1 + parse_retries,
                e,
                response.output_text,
            )
    raise last_error  # type: ignore[misc]


async def call_discriminator_with_messages(
    *,
    model: str,
    messages: list[dict[str, str]],
    is_reasoning: bool = False,
    reasoning_effort: str = "medium",
    temperature: float = 0.0,
    json_schema: dict | None = None,
    parse_retries: int = 1,
    max_retries: int = 3,
) -> tuple[dict[str, Any], list[dict[str, str]], LLMResponse]:
    """Call the discriminator and return the full message history.

    Calls call_llm_json() for JSON parsing with retries, then appends the
    assistant's response to the messages list. The returned messages can be
    passed to call_optimizer_with_messages() to give the optimizer full context.

    Returns:
        (parsed_json, messages_with_response, LLMResponse) tuple.
    """
    parsed, response = await call_llm_json(
        model=model,
        messages=messages,
        is_reasoning=is_reasoning,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        json_schema=json_schema,
        parse_retries=parse_retries,
        max_retries=max_retries,
    )

    # Append the assistant's response so the optimizer sees the full conversation
    messages_out = messages + [{"role": "assistant", "content": response.output_text}]
    return parsed, messages_out, response


async def call_optimizer_with_messages(
    *,
    model: str,
    messages: list[dict[str, str]],
    optimizer_prompt: str,
    is_reasoning: bool = False,
    reasoning_effort: str = "medium",
    temperature: float = 0.0,
    json_schema: dict | None = None,
    parse_retries: int = 1,
    max_retries: int = 3,
) -> tuple[dict[str, Any], LLMResponse]:
    """Call the optimizer with the full discriminator conversation as context.

    Takes the messages list returned by call_discriminator_with_messages()
    (which includes the system prompt, user prompt, and assistant response),
    appends the optimizer's user prompt, and makes the call. The optimizer
    sees the full judging context explicitly in the message history.

    Returns:
        (parsed_json, LLMResponse) tuple.
    """
    full_messages = messages + [{"role": "user", "content": optimizer_prompt}]
    return await call_llm_json(
        model=model,
        messages=full_messages,
        is_reasoning=is_reasoning,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        json_schema=json_schema,
        parse_retries=parse_retries,
        max_retries=max_retries,
    )
