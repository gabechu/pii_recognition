from collections import Counter
from typing import List
from unittest.mock import Mock

import numpy as np
import pytest
from pytest import fixture

from pii_recognition.labels.schema import EvalLabel, SpanLabel
from pii_recognition.tokenisation.token_schema import Token

from .model_evaluator import ModelEvaluator
from .prediction_error import SampleError, TokenError


@fixture
def text():
    return "This is Bob from Melbourne."


@fixture
def mock_recogniser():
    recogniser = Mock()
    recogniser.analyse.return_value = [
        SpanLabel("PER", 8, 11),
        SpanLabel("LOC", 17, 26),
    ]
    recogniser.supported_entities = ["PER", "LOC"]
    return recogniser


@fixture
def mock_bad_recogniser():
    # failed to predict location entity
    recogniser = Mock()
    recogniser.analyse.return_value = [
        SpanLabel("PER", 8, 11),
    ]
    recogniser.supported_entities = ["PER", "LOC"]
    return recogniser


@fixture
def mock_tokeniser():
    # reference: text = "This is Bob from Melbourne."
    tokeniser = Mock()
    tokeniser.tokenise.return_value = [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("Bob", 8, 11),
        Token("from", 12, 16),
        Token("Melbourne", 17, 26),
        Token(".", 26, 27),
    ]
    return tokeniser


def get_tokens() -> List:
    return ["This", "is", "Bob", "from", "Melbourne", "."]


def test_class_init():
    mock_recogniser = Mock()
    mock_recogniser.supported_entities = ["PER", "LOC"]
    mock_tokeniser = Mock()

    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
        switch_labels={"PER": "PERSON", "LOC": "LOCATION"},
    )

    assert evaluator.recogniser == mock_recogniser
    assert evaluator.tokeniser == mock_tokeniser
    assert evaluator.target_entities == ["PER", "LOC"]
    assert evaluator._switch_labels == {"PER": "PERSON", "LOC": "LOCATION"}
    assert evaluator._translated_entities == ["PERSON", "LOCATION"]


def test_get_span_based_prediction(mock_recogniser, mock_tokeniser, text):
    # test 1: succeed
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
    )
    actual = evaluator.get_span_based_prediction(text)
    assert actual == [
        SpanLabel(entity_type="PER", start=8, end=11),
        SpanLabel(entity_type="LOC", start=17, end=26),
    ]

    # test 2: raise assertion error
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser, tokeniser=mock_tokeniser, target_entities=["PER"],
    )
    with pytest.raises(AssertionError) as err:
        evaluator.get_span_based_prediction(text)
    assert str(err.value) == f"Predictions contain unasked entities ['LOC']"


def test_get_token_based_prediction(mock_recogniser, mock_tokeniser, text):
    # test 1: succeed
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
    )
    actual = evaluator.get_token_based_prediction(text)
    assert [x.entity_type for x in actual] == ["O", "O", "PER", "O", "LOC", "O"]

    # test 2: raise assertion error
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser, tokeniser=mock_tokeniser, target_entities=["PER"],
    )
    with pytest.raises(AssertionError) as err:
        evaluator.get_token_based_prediction(text)
    assert str(err.value) == f"Predictions contain unasked entities ['LOC']"


def test__compare_predicted_and_truth(text, mock_recogniser):
    # target_entities does not matter for this test
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser, tokeniser=Mock(), target_entities=["PER", "LOC"]
    )

    # test 1: predicted == truths
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
        get_tokens(),
        annotations=["O", "O", "PER", "O", "LOC", "O"],
        predictions=["O", "O", "PER", "O", "LOC", "O"],
    )
    assert counter == Counter(
        {EvalLabel("O", "O"): 4, EvalLabel("LOC", "LOC"): 1, EvalLabel("PER", "PER"): 1}
    )
    assert mistakes is None

    # test 2: predicted != truths and 2 mistakes were made
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
        get_tokens(),
        annotations=["O", "O", "PER", "O", "LOC", "O"],
        predictions=["LOC", "O", "PER", "O", "O", "O"],
    )
    assert counter == Counter(
        {
            EvalLabel("O", "O"): 3,
            EvalLabel("O", "LOC"): 1,
            EvalLabel("PER", "PER"): 1,
            EvalLabel("LOC", "O"): 1,
        }
    )
    assert mistakes == SampleError(
        token_errors=[
            TokenError("O", "LOC", "This"),
            TokenError("LOC", "O", "Melbourne"),
        ],
        full_text=text,
        failed=False,
    )

    # test 3: len(predicted) != len(truths) and evaluation on this text failed
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
        get_tokens(),
        annotations=["O", "O", "PER", "O", "LOC", "O"],
        predictions=["LOC", "O", "PER", "O", "O"],
    )
    assert counter == Counter()
    assert mistakes == SampleError(token_errors=[], full_text=text, failed=True)

    # test 4: comparison with entity mapping
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=Mock(),
        target_entities=["LOC", "PER"],
        switch_labels={"LOC": "LOCATION", "PER": "PERSON"},
    )
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
        get_tokens(),
        annotations=["O", "O", "PERSON", "O", "LOCATION", "O"],
        predictions=["O", "O", "PER", "O", "LOC", "O"],
    )
    assert counter == Counter(
        {
            EvalLabel("O", "O"): 4,
            EvalLabel("PERSON", "PERSON"): 1,
            EvalLabel("LOCATION", "LOCATION"): 1,
        }
    )
    assert mistakes is None


def test_evaluate_sample_no_label_conversion(text, mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
    )

    # test 1: simple straightforward and pass
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "O", "PER", "O", "LOC", "O"]
    )
    assert counter == Counter(
        {EvalLabel("O", "O"): 4, EvalLabel("LOC", "LOC"): 1, EvalLabel("PER", "PER"): 1}
    )
    assert mistakes is None

    # test 2: annotated labels not in target_entities
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "MISC", "PER", "O", "LOC", "MISC"]
    )
    assert counter == Counter(
        {EvalLabel("O", "O"): 4, EvalLabel("LOC", "LOC"): 1, EvalLabel("PER", "PER"): 1}
    )
    assert mistakes is None

    # test 3: len of annotations mismatch with len of predictions
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "MISC", "PER", "O", "LOC"]
    )
    assert counter == {}
    assert mistakes == SampleError(
        token_errors=[], full_text="This is Bob from Melbourne.", failed=True
    )

    # test 4: recogniser predicted on ["PER", "LOC"] but only asking for ["PER"]
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser, tokeniser=mock_tokeniser, target_entities=["PER"],
    )
    with pytest.raises(AssertionError) as err:
        counter, mistakes = evaluator.evaluate_sample(
            text, annotations=["O", "O", "PER", "O", "LOC", "O"]
        )
    assert str(err.value) == f"Predictions contain unasked entities ['LOC']"


def test_evaluate_sample_with_label_conversion(text, mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
        switch_labels={"PER": "I-PER", "LOC": "I-LOC"},
    )
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "I-MISC", "I-PER", "O", "I-LOC", "I-MISC"]
    )
    assert counter == Counter(
        {
            EvalLabel("O", "O"): 4,
            EvalLabel("I-LOC", "I-LOC"): 1,
            EvalLabel("I-PER", "I-PER"): 1,
        }
    )
    assert mistakes is None


def test_evaluate_sample_with_mistakes(text, mock_bad_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_bad_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
        switch_labels={"PER": "I-PER", "LOC": "I-LOC"},
    )
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "I-MISC", "I-PER", "O", "I-LOC", "I-MISC"]
    )
    assert mistakes == SampleError(
        token_errors=[TokenError(annotation="I-LOC", prediction="O", text="Melbourne")],
        full_text="This is Bob from Melbourne.",
        failed=False,
    )


def test_evaulate_all(text, mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
    )
    counters, mistakes = evaluator.evaluate_all(
        texts=[text] * 2, annotations=[["O", "O", "PER", "O", "LOC", "O"]] * 2
    )

    assert (
        counters
        == [
            Counter(
                {
                    EvalLabel("O", "O"): 4,
                    EvalLabel("LOC", "LOC"): 1,
                    EvalLabel("PER", "PER"): 1,
                }
            )
        ]
        * 2
    )
    assert mistakes == []


def test_calculate_score(mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
    )

    # test 1: LOC 0.=presion=recall
    counters = [
        Counter(
            {
                EvalLabel("O", "O"): 3,
                EvalLabel("O", "LOC"): 1,
                EvalLabel("LOC", "O"): 1,
                EvalLabel("PER", "PER"): 1,
            }
        )
    ]
    recall, precision, f1 = evaluator.calculate_score(counters)
    assert recall == {"PER": 1.0, "LOC": 0.0}
    assert precision == {"PER": 1.0, "LOC": 0.0}
    assert f1 == {"PER": 1.0, "LOC": np.nan}

    # test 2: multiple texts
    counters = [
        Counter(
            {
                EvalLabel("O", "O"): 4,
                EvalLabel("LOC", "LOC"): 1,
                EvalLabel("PER", "PER"): 1,
            }
        )
    ] * 2
    recall, precision, f1 = evaluator.calculate_score(counters)
    assert recall == {"PER": 1.0, "LOC": 1.0}
    assert precision == {"PER": 1.0, "LOC": 1.0}
    assert f1 == {"PER": 1.0, "LOC": 1.0}

    # test 3: with entity mapping
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        tokeniser=mock_tokeniser,
        target_entities=["PER", "LOC"],
        switch_labels={"LOC": "LOCATION", "PER": "PERSON"},
    )
    counters = [
        Counter(
            {
                EvalLabel("O", "O"): 4,
                EvalLabel("LOCATION", "LOCATION"): 1,
                EvalLabel("PERSON", "PERSON"): 1,
            }
        )
    ] * 2

    recall, precision, f1 = evaluator.calculate_score(counters)
    assert recall == {"PERSON": 1.0, "LOCATION": 1.0}
    assert precision == {"PERSON": 1.0, "LOCATION": 1.0}
    assert f1 == {"PERSON": 1.0, "LOCATION": 1.0}

    # test 4: with entity mapping and use recogniser labels
    recall, precision, f1 = evaluator.calculate_score(counters, use_test_labels=False)
    assert recall == {"PER": 1.0, "LOC": 1.0}
    assert precision == {"PER": 1.0, "LOC": 1.0}
    assert f1 == {"PER": 1.0, "LOC": 1.0}
