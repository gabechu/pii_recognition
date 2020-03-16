from collections import Counter

import numpy as np
import pytest
from mock import Mock
from pytest import fixture

from label.label_schema import EvalLabel, SpanLabel
from tokeniser.token import Token

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
    return recogniser


@fixture
def mock_tokeniser():
    tokeniser = Mock()
    tokeniser.return_value = [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("Bob", 8, 11),
        Token("from", 12, 16),
        Token("Melbourne", 17, 26),
        Token(".", 26, 27),
    ]
    return tokeniser


def test_get_token_based_prediction(text, mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=mock_tokeniser,
    )
    actual = evaluator.get_token_based_prediction(text)
    assert actual == ["O", "O", "PER", "O", "LOC", "O"]

    evaluator = ModelEvaluator(
        recogniser=mock_recogniser, target_entities=["PER"], tokeniser=mock_tokeniser,
    )
    with pytest.raises(AssertionError) as err:
        evaluator.get_token_based_prediction(text)
    assert str(err.value) == f"Predictions contain unasked entities ['LOC']"


def test__compare_predicted_and_truth(text, mock_tokeniser):
    # test 1: predicted == truths
    evaluator = ModelEvaluator(
        recogniser=Mock(), target_entities=["ANY"], tokeniser=mock_tokeniser,
    )
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
        annotations=["O", "O", "PER", "O", "LOC", "O"],
        predictions=["O", "O", "PER", "O", "LOC", "O"],
    )
    assert counter == Counter(
        {EvalLabel("O", "O"): 4, EvalLabel("LOC", "LOC"): 1, EvalLabel("PER", "PER"): 1}
    )
    assert mistakes == SampleError(token_errors=[], full_text=text, failed=False)

    # test 2: predicted != truths where 2 mistakes were made
    evaluator = ModelEvaluator(
        recogniser=Mock(), target_entities=["ANY"], tokeniser=mock_tokeniser,
    )
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
        annotations=["O", "O", "PER", "O", "LOC", "O"],
        predictions=["LOC", "O", "PER", "O", "O", "O"],
    )
    assert counter == Counter(
        {
            EvalLabel("O", "O"): 3,
            EvalLabel("O", "LOC"): 1,
            EvalLabel("LOC", "O"): 1,
            EvalLabel("PER", "PER"): 1,
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

    # test 3: len(predicted) != len(truths)
    evaluator = ModelEvaluator(
        recogniser=Mock(), target_entities=["ANY"], tokeniser=mock_tokeniser
    )
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
        annotations=["O", "O", "PER", "O", "LOC", "O"],
        predictions=["LOC", "O", "PER", "O", "O"],
    )
    assert counter == Counter()
    assert mistakes == SampleError(token_errors=[], full_text=text, failed=True)

    # test 4: entity mapping
    evaluator = ModelEvaluator(
        recogniser=Mock(),
        target_entities=["ANY"],
        tokeniser=mock_tokeniser,
        entity_mapping={"LOC": "LOCATION", "PER": "PERSON"},
    )
    counter, mistakes = evaluator._compare_predicted_and_truth(
        text,
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
    assert mistakes == SampleError(token_errors=[], full_text=text, failed=False)


def test_evaluate_sample(text, mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=mock_tokeniser,
    )
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "O", "PER", "O", "LOC", "O"]
    )
    assert counter == Counter(
        {EvalLabel("O", "O"): 4, EvalLabel("LOC", "LOC"): 1, EvalLabel("PER", "PER"): 1}
    )
    assert mistakes == SampleError(token_errors=[], full_text=text, failed=False)


def test_evaulate_all(text, mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=mock_tokeniser,
    )
    counters, mistakes = evaluator.evaulate_all(
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
    assert mistakes == [SampleError(token_errors=[], full_text=text, failed=False)] * 2


def test_calculate_score():
    evaluator = ModelEvaluator(
        recogniser=Mock(), target_entities=["PER", "LOC"], tokeniser=Mock()
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
    actual = evaluator.calculate_score(counters)
    assert actual == {"PER": 1.0, "LOC": np.nan}

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
    actual = evaluator.calculate_score(counters)
    assert actual == {"PER": 1.0, "LOC": 1.0}

    # test 3: with entity mapping
    evaluator = ModelEvaluator(
        recogniser=Mock(),
        target_entities=["PER", "LOC"],
        tokeniser=Mock(),
        entity_mapping={"LOC": "LOCATION", "PER": "PERSON"},
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
    actual = evaluator.calculate_score(counters)
    assert actual == {"PERSON": 1.0, "LOCATION": 1.0}
