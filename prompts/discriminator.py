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
    return f"\nCRITERIA TO EVALUATE ON:\n{bullets}\n"


def _format_constraints_block(constraints: str | list[str]) -> str:
    """Format constraints as a bulleted block."""
    if not constraints:
        return ""
    if isinstance(constraints, str):
        items = [constraints]
    else:
        items = constraints
    bullets = "\n".join(f"- {item}" for item in items)
    return f"\nADDITIONAL INFORMATION:\n{bullets}\n"


def build_instruction_trajectory(
    outputs: list[ModelOutput],
    samples: list[Sample],
    show_target: bool,
) -> str:
    """Build trajectory for instruction mode.

    Format per sample:
        Question: {sample.question}
        Response: {output.response}
        Expected Answer: {sample.target}   # Only if show_target=True
    """
    sample_map = {s.index: s for s in samples}
    lines: list[str] = []
    for i, output in enumerate(outputs):
        sample = sample_map[output.sample_index]
        lines.append(f"--- Sample {i + 1} ---")
        lines.append(f"Question:\n{sample.question}")
        lines.append(f"Response:\n{output.response}")
        if show_target:
            lines.append(f"Expected Answer:\n{sample.target}")
    return "\n".join(lines)


def build_standalone_trajectory(
    outputs: list[ModelOutput],
    grader: Grader,
    show_target: bool,
    prompt_text: str | None = None,
) -> str:
    """Build trajectory for standalone mode.

    Format per output:
        Output: {output.response}
        Grade: {grader.check_output(output.response, prompt_text)}  # Only if show_target=True

    Raises ValueError if show_target=True and check_output() returns None.
    """
    lines: list[str] = []
    for i, output in enumerate(outputs):
        lines.append(f"--- Output {i + 1} ---")
        lines.append(f"Output:\n{output.response}")
        if show_target:
            annotation = grader.check_output(output.response, prompt_text)
            if annotation is None:
                raise ValueError(
                    "show_target=True requires grader.check_output() to return "
                    "a dict, but it returned None. Override check_output() in "
                    "your Grader subclass."
                )
            lines.append(f"Grade: {annotation}")
    return "\n".join(lines)


def build_discriminator_prompt(
    trajectory_a: str,
    trajectory_b: str,
    config: DiscriminatorConfig,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the discriminator.

    The system prompt is a short role description. call_llm() handles
    the system -> developer mapping for reasoning models internally.
    """
    system_prompt = (
        "You are an evaluator of LLMs. You will be given examples of outputs "
        "from the same LLM with two different instructions. You must choose "
        "the version you prefer, based off of the evaluation criteria provided. "
        "Then provide feedback about why you chose that one "
        "and what can be improved about the one you didn't choose. Then, you will "
        "be given the non-preferred instruction and the feedback, and you will be "
        "asked to produce an improved instruction based on the feedback."
    )

    parts: list[str] = []
    parts.append(f"<Version 1>\n{trajectory_a}\n</Version 1>")
    parts.append(f"\n<Version 2>\n{trajectory_b}\n</Version 2>")

    criteria_block = _format_criteria_block(config.criteria)
    if criteria_block:
        parts.append(criteria_block)

    constraints_block = _format_constraints_block(config.constraints)
    if constraints_block:
        parts.append(constraints_block)

    parts.append(
        "\n<Task>Be very smart, logical, and critical. Just provide concise "
        "feedback. First do your best to reason about what "
        "is the ideal behavior given the evaluation criteria and choose the responses that align most with "
        "this. Then, provide clear, generalizable feedback that doesn't rely on "
        "the specific responses, instead discuss why you chose that one "
        "and what can be improved about the one you didn't choose.</Task>"
    )

    parts.append(
        '\n<Output>The output should be a JSON object with the following fields: '
        '"preferred": 1 or 2, "feedback": string.</Output>'
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
