"""Utility for formatting prompts and calling the task model concurrently."""

import asyncio

from prefpo.config import ModelConfig
from prefpo.llm.client import call_llm
from prefpo.types import ModelOutput, Prompt, PromptRole, Sample


def _format_prompt_sent(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)


def format_instruction_messages(
    prompt: Prompt, sample: Sample
) -> list[dict[str, str]]:
    """Build the message list for instruction mode (prompt + question).

    USER role: single user message, instruction prepended to question.
    SYSTEM role: instruction in system message, question in user message.
    Empty prompt: omit instruction, just send question.
    """
    if prompt.value == "":
        return [{"role": "user", "content": sample.question}]

    if prompt.role == PromptRole.USER:
        return [{"role": "user", "content": f"{prompt.value}\n\n{sample.question}"}]
    else:
        return [
            {"role": "system", "content": prompt.value},
            {"role": "user", "content": sample.question},
        ]


def format_standalone_messages(prompt: Prompt) -> list[dict[str, str]]:
    """Build the message list for standalone mode (prompt only).

    Standalone mode requires USER role — the prompt IS the user input.
    """
    if prompt.role != PromptRole.USER:
        raise ValueError("Standalone mode requires prompt_role='user'")
    return [{"role": "user", "content": prompt.value}]


async def generate_outputs(
    prompt: Prompt,
    samples: list[Sample],
    model_config: ModelConfig,
    semaphore: asyncio.Semaphore,
) -> list[ModelOutput]:
    """Instruction mode: run the task model on every sample concurrently.

    Uses shared semaphore + asyncio.gather() pattern.
    """

    async def _generate_one(sample: Sample) -> ModelOutput:
        messages = format_instruction_messages(prompt, sample)
        async with semaphore:
            response = await call_llm(
                model=model_config.name,
                messages=messages,
                is_reasoning=model_config.is_reasoning,
                reasoning_effort=model_config.reasoning_effort,
                temperature=model_config.temperature,
            )
        return ModelOutput(
            sample_index=sample.index,
            prompt_sent=_format_prompt_sent(messages),
            response=response.output_text,
        )

    results = await asyncio.gather(*[_generate_one(s) for s in samples])
    return list(results)


async def generate_standalone(
    prompt: Prompt,
    model_config: ModelConfig,
    semaphore: asyncio.Semaphore,
    n: int = 1,
) -> list[ModelOutput]:
    """Standalone mode: run the task model on the prompt directly.

    Generates n outputs from the same prompt. Returns ModelOutput with
    sample_index=-1 (no associated sample).
    """
    messages = format_standalone_messages(prompt)

    async def _generate_one(i: int) -> ModelOutput:
        async with semaphore:
            response = await call_llm(
                model=model_config.name,
                messages=messages,
                is_reasoning=model_config.is_reasoning,
                reasoning_effort=model_config.reasoning_effort,
                temperature=model_config.temperature,
            )
        return ModelOutput(
            sample_index=-1,
            prompt_sent=_format_prompt_sent(messages),
            response=response.output_text,
        )

    results = await asyncio.gather(*[_generate_one(i) for i in range(n)])
    return list(results)
