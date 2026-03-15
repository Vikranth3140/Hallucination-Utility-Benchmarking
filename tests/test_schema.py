import pytest
from pydantic import ValidationError
from src.schema import LabeledExample

def test_valid_unsupported_0():
    ex = LabeledExample(
        task_type="factual",
        prompt_id="p1",
        prompt="Who is the president?",
        model_name="llama",
        response="I don't know.",
        unsupported_label=0,
        utility_label=None,
        rationale=""
    )
    assert ex.unsupported_label == 0
    assert ex.utility_label is None

def test_valid_unsupported_1():
    ex = LabeledExample(
        task_type="creative",
        prompt_id="p2",
        prompt="Write a poem.",
        model_name="mistral",
        response="Here is a poem...",
        unsupported_label=1,
        utility_label="useful",
        rationale="Useful hallucination"
    )
    assert ex.unsupported_label == 1
    assert ex.utility_label == "useful"

def test_invalid_unsupported_0_with_utility():
    with pytest.raises(ValidationError):
        LabeledExample(
            task_type="factual",
            prompt_id="p1",
            prompt="Test",
            model_name="llama",
            response="Test response",
            unsupported_label=0,
            utility_label="neutral"
        )

def test_invalid_unsupported_1_without_utility():
    with pytest.raises(ValidationError):
        LabeledExample(
            task_type="factual",
            prompt_id="p1",
            prompt="Test",
            model_name="llama",
            response="Test response",
            unsupported_label=1,
            utility_label=None
        )

def test_invalid_unsupported_1_wrong_utility():
    with pytest.raises(ValidationError):
        LabeledExample(
            task_type="factual",
            prompt_id="p1",
            prompt="Test",
            model_name="llama",
            response="Test response",
            unsupported_label=1,
            # using an old label here to test rejection
            utility_label="U+" 
        )

def test_invalid_unsupported_value():
    with pytest.raises(ValidationError):
        LabeledExample(
            task_type="factual",
            prompt_id="p1",
            prompt="Test",
            model_name="llama",
            response="Test response",
            unsupported_label=2,
            utility_label=None
        )
