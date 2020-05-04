from collections import Counter
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pytest import fixture

from pii_recognition.labels.schema import EvalLabel, SpanLabel
from pii_recognition.tokenisation.token_schema import Token

from .model_evaluator import ModelEvaluator, tokeniser_registry
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
def tokeniser_config():
    return {"name": "fake_tokeniser", "config": {"fake_param": "fake_value"}}


def get_mock_tokeniser():
    # reference: text = "This is Bob from Melbourne."
    tokeniser = Mock()
    tokeniser.return_value.tokenise.return_value = [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("Bob", 8, 11),
        Token("from", 12, 16),
        Token("Melbourne", 17, 26),
        Token(".", 26, 27),
    ]
    return tokeniser


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_class_init(mock_tokeniser, tokeniser_config):
    mock_recogniser = Mock()

    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=tokeniser_config,
        convert_labels={"PER": "PERSON"},
    )

    assert evaluator.recogniser == mock_recogniser
    assert evaluator.target_entities == ["PER", "LOC"]
    mock_tokeniser.assert_called_with("fake_tokeniser", {"fake_param": "fake_value"})
    assert evaluator._convert_labels == {"PER": "PERSON"}


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_get_token_based_prediction(
    mock_tokeniser, text, mock_recogniser, tokeniser_config
):
    # test 1: succeed
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=tokeniser_config,
    )
    actual = evaluator.get_token_based_prediction(text)
    assert [x.entity_type for x in actual] == ["O", "O", "PER", "O", "LOC", "O"]

    # test 2: raise assertion error
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser, target_entities=["PER"], tokeniser=tokeniser_config,
    )
    with pytest.raises(AssertionError) as err:
        evaluator.get_token_based_prediction(text)
    assert str(err.value) == f"Predictions contain unasked entities ['LOC']"


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test__compare_predicted_and_truth(mock_tokeniser, text, tokeniser_config):
    # test 1: predicted == truths
    evaluator = ModelEvaluator(
        recogniser=Mock(), target_entities=["ANY"], tokeniser=tokeniser_config,
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
        recogniser=Mock(), target_entities=["ANY"], tokeniser=tokeniser_config,
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
        recogniser=Mock(), target_entities=["ANY"], tokeniser=tokeniser_config
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
        tokeniser=tokeniser_config,
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


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_evaluate_sample(mock_tokeniser, text, mock_recogniser, tokeniser_config):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=tokeniser_config,
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


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_evaluate_sample_with_label_conversion(
    mock_tokeniser, mock_recogniser, tokeniser_config
):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=tokeniser_config,
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


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_evaulate_all(mock_tokeniser, text, mock_recogniser, tokeniser_config):
    evaluator = ModelEvaluator(
        recogniser=mock_recogniser,
        target_entities=["PER", "LOC"],
        tokeniser=tokeniser_config,
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


@patch.object(
    target=tokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_tokeniser,
)
def test_calculate_score(mock_tokeniser, tokeniser_config):
    evaluator = ModelEvaluator(
        recogniser=Mock(), target_entities=["PER", "LOC"], tokeniser=tokeniser_config
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
        tokeniser=tokeniser_config,
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
