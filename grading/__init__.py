"""Grading system for PrefPO — grader implementations and factory."""

from prefpo.data.bbh import BINARY_TASKS, EXACT_MATCH_TASKS, MULTIPLE_CHOICE_TASKS
from prefpo.grading.base import GradeResult, Grader
from prefpo.grading.binary import BinaryGrader
from prefpo.grading.exact_match import ExactMatchGrader
from prefpo.grading.ifeval import IFEvalGrader
from prefpo.grading.multiple_choice import MultipleChoiceGrader


def get_bbh_grader(subset_name: str) -> Grader:
    """Return the appropriate grader for a BBH subtask."""
    if subset_name in MULTIPLE_CHOICE_TASKS:
        return MultipleChoiceGrader()
    if subset_name in BINARY_TASKS:
        return BinaryGrader()
    if subset_name in EXACT_MATCH_TASKS:
        return ExactMatchGrader()
    raise ValueError(
        f"Unknown BBH subset: {subset_name}. "
        f"Known: {sorted(MULTIPLE_CHOICE_TASKS | set(BINARY_TASKS) | set(EXACT_MATCH_TASKS))}"
    )


__all__ = [
    "Grader",
    "GradeResult",
    "MultipleChoiceGrader",
    "BinaryGrader",
    "ExactMatchGrader",
    "IFEvalGrader",
    "get_bbh_grader",
]
