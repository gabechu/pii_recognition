from collections import Counter
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


def test_class_init():
    mock_recogniser = Mock()
    mock_tokeniser = Mock()

    # test 1: succeed
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=mock_tokeniser,
        convert_labels={"PER": "PERSON"},
    )

    assert evaluator.recogniser == mock_recogniser
    assert evaluator.target_entities == ["PER", "LOC"]
    assert evaluator.tokeniser == mock_tokeniser
    assert evaluator.convert_labels == {"PER": "PERSON"}

    # test 2: raise assertion error
    with pytest.raises(AssertionError) as err:
        evaluator = ModelEvaluator(
            recogniser=mock_recogniser,
            target_entities=["PER", "PER", "LOC"],
            tokeniser=mock_tokeniser,
        )
    assert (
        str(err.value)
        == "No repeated entities are allowed, but found ['PER', 'PER', 'LOC']."
    )


def test_get_token_based_prediction(text, mock_recogniser, mock_tokeniser):
    # test 1: succeed
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=mock_tokeniser,
    )
    actual = evaluator.get_token_based_prediction(text)
    assert actual == ["O", "O", "PER", "O", "LOC", "O"]

    # test 2: raise assertion error
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
        convert_labels={"LOC": "LOCATION", "PER": "PERSON"},
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

    # test 1: simple straightforward pass
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "O", "PER", "O", "LOC", "O"]
    )
    assert counter == Counter(
        {EvalLabel("O", "O"): 4, EvalLabel("LOC", "LOC"): 1, EvalLabel("PER", "PER"): 1}
    )
    assert mistakes == SampleError(token_errors=[], full_text=text, failed=False)

    # test 2: annotated labels not in target_entities
    counter, mistakes = evaluator.evaluate_sample(
        text, annotations=["O", "MISC", "PER", "O", "LOC", "MISC"]
    )
    assert counter == Counter(
        {EvalLabel("O", "O"): 4, EvalLabel("LOC", "LOC"): 1, EvalLabel("PER", "PER"): 1}
    )
    assert mistakes == SampleError(token_errors=[], full_text=text, failed=False)


def test_evaluate_sample_with_label_conversion(mock_recogniser, mock_tokeniser):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=mock_tokeniser,
        convert_labels={"PER": "I-PER", "LOC": "I-LOC"},
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
        recogniser=Mock(),
        target_entities=["PER", "LOC"],
        tokeniser=Mock(),
        convert_labels={"LOC": "LOCATION", "PER": "PERSON"},
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
