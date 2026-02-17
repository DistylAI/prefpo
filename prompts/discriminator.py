"""Discriminator prompt builder — builds comparison prompts from trajectories."""

from prefpo.config import DiscriminatorConfig
from prefpo.grading.base import Grader
from prefpo.types import ModelOutput, Sample


def _format_criteria_block(criteria: str | list[str]) -> str:
    """Format criteria as a bulleted block."""
    if not criteria:
        return ""
    if isinstance(criteria, str):
        items = [criteria]
    else:
        items = criteria
    bullets = "\n".join(f"- {item}" for item in items)
    return f"\n<Criteria to Evaluate On>\n{bullets}\n</Criteria to Evaluate On>"


def _format_additional_info_block(additional_info: str | list[str]) -> str:
    """Format additional info as a bulleted block."""
    if not additional_info:
        return ""
    if isinstance(additional_info, str):
        items = [additional_info]
    else:
        items = additional_info
    bullets = "\n".join(f"- {item}" for item in items)
    return f"\n<Additional Information>\n{bullets}\n</Additional Information>"


def build_instruction_trajectory(
    outputs: list[ModelOutput],
    samples: list[Sample],
    show_expected: bool,
) -> str:
    """Build trajectory for instruction mode.

    Format per sample:
        Question: {sample.question}
        Response: {output.response}
        Expected Answer: {sample.target}   # Only if show_expected=True
    """
    sample_map = {s.index: s for s in samples}
    lines: list[str] = []
    for i, output in enumerate(outputs):
        sample = sample_map[output.sample_index]
        lines.append(f"--- Sample {i + 1} ---")
        lines.append(f"Question:\n{sample.question}")
        lines.append(f"Response:\n{output.response}")
        if show_expected:
            lines.append(f"Expected Answer:\n{sample.target}")
    return "\n".join(lines)


def build_standalone_trajectory(
    outputs: list[ModelOutput],
    grader: Grader,
    show_expected: bool,
    prompt_text: str | None = None,
) -> str:
    """Build trajectory for standalone mode.

    Format per output:
        Output: {output.response}
        Grade: {grader.check_output(output.response, prompt_text)}  # Only if show_expected=True

    Raises ValueError if show_expected=True and check_output() returns None.
    """
    lines: list[str] = []
    for i, output in enumerate(outputs):
        lines.append(f"--- Output {i + 1} ---")
        lines.append(f"Output:\n{output.response}")
        if show_expected:
            annotation = grader.check_output(output.response, prompt_text)
            if annotation is None:
                raise ValueError(
                    "show_expected=True requires grader.check_output() to return "
                    "a dict, but it returned None. Override check_output() in "
                    "your Grader subclass."
                )
            lines.append(f"Grade:\n{annotation}")
    return "\n".join(lines)


def build_discriminator_prompt(
    trajectory_a: str,
    trajectory_b: str,
    config: DiscriminatorConfig,
    mode: str = "instruction",
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the discriminator.

    The system prompt is a short role description. call_llm() handles
    the system -> developer mapping for reasoning models internally.
    """
    if mode == "instruction":
        mode_detail = (
            " For each sample, you will see the question, the model's "
            "response, and optionally the expected answer."
        )
    else:
        mode_detail = (
            " You will see the model's output and "
            "optionally a grade from an automated checker."
        )

    system_prompt = (
        "You are an evaluator of LLMs. You will be given examples of outputs "
        "from the same LLM with two different instructions."
        f"{mode_detail} You must choose "
        "the version you prefer, based off of the evaluation criteria provided. "
        "Then provide feedback about why you chose that one "
        "and what can be improved about the one you didn't choose. Then, you will "
        "be given the non-preferred instruction and the feedback, and you will be "
        "asked to produce an improved instruction based on the feedback."
    )

    parts: list[str] = []
    parts.append(f"<Version 1>\n{trajectory_a}\n</Version 1>")
    parts.append(f"\n<Version 2>\n{trajectory_b}\n</Version 2>")

    info_block = _format_additional_info_block(config.additional_info)
    if info_block:
        parts.append(info_block)

    criteria_block = _format_criteria_block(config.criteria)
    if criteria_block:
        parts.append(criteria_block)

    parts.append(
        "\n<Task>\nBe very smart, logical, and critical. Just provide concise "
        "feedback. First do your best to reason about what "
        "is the ideal behavior given the evaluation criteria and choose the responses that align most with "
        "this. Then, provide clear, generalizable feedback that doesn't rely on "
        "the specific responses, instead discuss why you chose that one "
        "and what can be improved about the one you didn't choose.\n</Task>"
    )

    parts.append(
        '\n<Output>\nThe output should be a JSON object with the following fields: '
        '"preferred": 1 or 2, "feedback": string.\n</Output>'
    )

    user_prompt = "\n".join(parts)
    return system_prompt, user_prompt


DISCRIMINATOR_SCHEMA = {
    "type": "json_schema",
    "name": "discriminator_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "preferred": {"type": "integer", "enum": [1, 2]},
            "feedback": {"type": "string"},
        },
        "required": ["preferred", "feedback"],
        "additionalProperties": False,
    },
}
