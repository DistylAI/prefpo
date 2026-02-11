"""Tests for prefpo.grading.ifeval and prefpo.data.ifeval."""

from prefpo.config import PrefPOConfig
from prefpo.data.ifeval import build_ifeval_config, load_ifeval_dataset, load_ifeval_sample
from prefpo.grading.ifeval import IFEvalGrader, get_human_readable_criteria
# --- get_human_readable_criteria ---
def test_human_readable_criteria_keywords():
    """Known instruction_id returns descriptive text."""
    result = get_human_readable_criteria(
        "keywords:existence",
        {"keywords": ["hello", "world"]},
    )
    assert "hello" in result
    assert "world" in result
    assert "keywords" in result.lower()
def test_human_readable_criteria_unknown():
    """Unknown id returns 'Unknown instruction: ...'."""
    result = get_human_readable_criteria("nonexistent:instruction", {})
    assert result.startswith("Unknown instruction:")
def test_human_readable_criteria_all_types():
    """All 28 instruction types produce non-empty string."""
    known_types = [
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "combination:repeat_prompt",
        "combination:two_responses",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:constrained_response",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:title",
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
        "keywords:letter_frequency",
        "language:response_language",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences",
        "length_constraints:number_words",
        "punctuation:no_comma",
        "startend:end_checker",
        "startend:quotation",
    ]
    for inst_id in known_types:
        result = get_human_readable_criteria(inst_id, {})
        assert len(result) > 0, f"Empty result for {inst_id}"
        assert not result.startswith("Unknown"), f"Unknown for {inst_id}"
# --- IFEvalGrader.check_output ---
def test_ifeval_grader_check_output():
    """check_output returns dict with passed_strict, passed_loose, inst_strict."""
    grader = IFEvalGrader(
        instruction_id_list=["punctuation:no_comma"],
        kwargs=[{}],
        n_eval_trials=1,
    )
    result = grader.check_output("This response has no commas at all", "Write something")
    assert "passed_strict" in result
    assert "passed_loose" in result
    assert "inst_strict" in result
    assert isinstance(result["passed_strict"], bool)
# --- Data loader tests ---
def test_load_ifeval_sample():
    """load_ifeval_sample returns valid sample dict."""
    sample = load_ifeval_sample(0)
    assert "prompt" in sample
    assert "instruction_id_list" in sample
    assert "kwargs" in sample
    assert "criteria" in sample
    assert isinstance(sample["criteria"], list)
    assert len(sample["criteria"]) > 0
def test_load_ifeval_dataset():
    """load_ifeval_dataset returns full dataset."""
    dataset = load_ifeval_dataset()
    assert len(dataset) > 0
    assert "prompt" in dataset[0]
# --- build_ifeval_config ---
def test_build_ifeval_config_no_base():
    """Creates standalone config with correct defaults."""
    sample = load_ifeval_sample(0)
    config, grader = build_ifeval_config(sample)
    assert config.mode == "standalone"
    assert config.pool.prompt_role == "user"
    assert config.discriminator.show_expected is True
    assert isinstance(grader, IFEvalGrader)
def test_build_ifeval_config_with_base():
    """Overrides base config fields."""
    sample = load_ifeval_sample(0)
    base = PrefPOConfig(
        mode="instruction",
        task_model={"name": "gpt-4o-mini"},
        pool={"initial_prompts": ["placeholder"]},
        run={"iterations": 3},
    )
    config, grader = build_ifeval_config(sample, base_config=base)
    assert config.mode == "standalone"
    assert config.task_model.name == "gpt-4o-mini"
    assert config.run.iterations == 3
    assert config.pool.initial_prompts == [sample["prompt"]]
def test_build_ifeval_config_sets_show_expected():
    """Always sets show_expected=True."""
    sample = load_ifeval_sample(0)
    base = PrefPOConfig(
        mode="standalone",
        task_model={"name": "openai/gpt-4o"},
        discriminator={"show_expected": False},
        pool={"initial_prompts": ["placeholder"]},
    )
    config, _ = build_ifeval_config(sample, base_config=base)
    assert config.discriminator.show_expected is True
