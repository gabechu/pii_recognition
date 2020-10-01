from typing import List
from numpy.testing import assert_almost_equal

import pytest
from pii_recognition.evaluation.character_level_evaluation import (
    compute_entity_precisions_for_prediction,
    compute_entity_recalls_for_ground_truth,
    compute_pii_detection_f1,
    label_encoder,
)
from pii_recognition.labels.schema import Entity


def test_label_encoder_for_reserved_0_taken():
    with pytest.raises(ValueError) as err:
        label_encoder(3, [], {"LOC": 0})
    assert str(err.value) == (
        "Value 0 is reserved! If a character does not "
        "belong to any entity, it would be assigned with 0."
    )


def test_label_encoder_for_multi_labels():
    spans = [
        Entity(entity_type="LOC", start=5, end=8),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="PERSON", start=2, end=5),
    ]

    # entity PER and PERSON map to the same int
    actual = label_encoder(20, spans, {"LOC": 1, "PER": 2, "PERSON": 2})
    assert actual == [0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]


def test_label_encoder_for_missing_label_in_mapping():
    spans = [
        Entity(entity_type="LOC", start=5, end=8),
        Entity(entity_type="PER", start=10, end=15),
    ]

    with pytest.raises(Exception) as error:
        label_encoder(20, spans, {"LOC": 1})
    assert str(error.value) == ("Missing label 'PER' in 'label_to_int' mapping.")


def test_label_encoder_for_span_beyond_range():
    spans = [Entity(entity_type="LOC", start=3, end=7)]

    with pytest.raises(ValueError) as error:
        label_encoder(5, spans, {"LOC": 1})
    assert str(error.value) == (
        "Entity span index is out of range: text length is 5 but got span index 7."
    )


def test_compute_precisions_recalls_for_exact_match():
    true_entities = pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="LOC", start=23, end=32),
        Entity(entity_type="PER", start=37, end=48),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 3, "end": 7, "precision": 1.0},
        {"entity_type": "PER", "start": 10, "end": 15, "precision": 1.0},
        {"entity_type": "LOC", "start": 23, "end": 32, "precision": 1.0},
        {"entity_type": "PER", "start": 37, "end": 48, "precision": 1.0},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 1.0},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 1.0},
        {"entity_type": "LOC", "start": 23, "end": 32, "recall": 1.0},
        {"entity_type": "PER", "start": 37, "end": 48, "recall": 1.0},
    ]


def test_compute_precisions_recalls_for_pred_subset_of_true():
    # Every predicted entity is a subset of one true entity
    true_entities = pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="LOC", start=23, end=32),
        Entity(entity_type="PER", start=37, end=48),
    ]
    pred_entities = [
        Entity(entity_type="LOC", start=4, end=7),
        Entity(entity_type="PER", start=13, end=15),
        Entity(entity_type="LOC", start=25, end=30),
        Entity(entity_type="PER", start=40, end=46),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 4, "end": 7, "precision": 1.0},
        {"entity_type": "PER", "start": 13, "end": 15, "precision": 1.0},
        {"entity_type": "LOC", "start": 25, "end": 30, "precision": 1.0},
        {"entity_type": "PER", "start": 40, "end": 46, "precision": 1.0},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 0.75},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 0.4},
        {"entity_type": "LOC", "start": 23, "end": 32, "recall": 5 / 9},
        {"entity_type": "PER", "start": 37, "end": 48, "recall": 6 / 11},
    ]


def test_compute_precisions_recalls_for_pred_superset_of_true():
    # Every predicted entity is a super-set of one true entity
    true_entities = [
        Entity(entity_type="LOC", start=4, end=7),
        Entity(entity_type="PER", start=13, end=15),
        Entity(entity_type="LOC", start=25, end=30),
        Entity(entity_type="PER", start=40, end=46),
    ]
    pred_entities = pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="LOC", start=23, end=32),
        Entity(entity_type="PER", start=37, end=48),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 3, "end": 7, "precision": 0.75},
        {"entity_type": "PER", "start": 10, "end": 15, "precision": 0.4},
        {"entity_type": "LOC", "start": 23, "end": 32, "precision": 5 / 9},
        {"entity_type": "PER", "start": 37, "end": 48, "precision": 6 / 11},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 4, "end": 7, "recall": 1.0},
        {"entity_type": "PER", "start": 13, "end": 15, "recall": 1.0},
        {"entity_type": "LOC", "start": 25, "end": 30, "recall": 1.0},
        {"entity_type": "PER", "start": 40, "end": 46, "recall": 1.0},
    ]


def test_compute_precisions_recalls_for_pred_overlap_true():
    # Every predicted entity overlaps with one true entity
    true_entities = pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="LOC", start=23, end=32),
        Entity(entity_type="PER", start=37, end=48),
    ]
    pred_entities = [
        Entity(entity_type="LOC", start=1, end=4),
        Entity(entity_type="PER", start=13, end=18),
        Entity(entity_type="LOC", start=28, end=35),
        Entity(entity_type="PER", start=45, end=49),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 1, "end": 4, "precision": 1 / 3},
        {"entity_type": "PER", "start": 13, "end": 18, "precision": 0.4},
        {"entity_type": "LOC", "start": 28, "end": 35, "precision": 4 / 7},
        {"entity_type": "PER", "start": 45, "end": 49, "precision": 0.75},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 0.25},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 0.4},
        {"entity_type": "LOC", "start": 23, "end": 32, "recall": 4 / 9},
        {"entity_type": "PER", "start": 37, "end": 48, "recall": 3 / 11},
    ]


def test_compute_precisions_recalls_for_no_overlap():
    # No predicted entity overlaps with any true entity
    true_entities = pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="LOC", start=23, end=32),
        Entity(entity_type="PER", start=37, end=48),
    ]
    pred_entities = [
        Entity(entity_type="LOC", start=15, end=20),
        Entity(entity_type="PER", start=33, end=35),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 15, "end": 20, "precision": 0.0},
        {"entity_type": "PER", "start": 33, "end": 35, "precision": 0.0},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 0.0},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 0.0},
        {"entity_type": "LOC", "start": 23, "end": 32, "recall": 0.0},
        {"entity_type": "PER", "start": 37, "end": 48, "recall": 0.0},
    ]


def test_compute_precisions_recalls_for_one_pred_to_many_trues():
    # Every predicted entity overlaps more than one true entities
    # but its type may not match with all overlapping true entities
    true_entities = pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="LOC", start=23, end=32),
        Entity(entity_type="PER", start=37, end=48),
    ]
    pred_entities = [
        Entity(entity_type="LOC", start=3, end=20),
        Entity(entity_type="PER", start=28, end=43),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 3, "end": 20, "precision": 4 / 17},
        {"entity_type": "PER", "start": 28, "end": 43, "precision": 0.4},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 1.0},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 0.0},
        {"entity_type": "LOC", "start": 23, "end": 32, "recall": 0.0},
        {"entity_type": "PER", "start": 37, "end": 48, "recall": 6 / 11},
    ]


def test_compute_precisions_recalls_for_many_preds_to_one_true():
    # Every true entity overlaps more than one predicted entities
    # but its type may not match with all overlapping predicted entities
    true_entities = [
        Entity(entity_type="LOC", start=3, end=20),
        Entity(entity_type="PER", start=28, end=43),
    ]
    pred_entities = pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
        Entity(entity_type="LOC", start=23, end=32),
        Entity(entity_type="PER", start=37, end=48),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 3, "end": 7, "precision": 1.0},
        {"entity_type": "PER", "start": 10, "end": 15, "precision": 0.0},
        {"entity_type": "LOC", "start": 23, "end": 32, "precision": 0.0},
        {"entity_type": "PER", "start": 37, "end": 48, "precision": 6 / 11},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 20, "recall": 4 / 17},
        {"entity_type": "PER", "start": 28, "end": 43, "recall": 0.4},
    ]


def test_compute_precisions_recalls_for_incorrect_type():
    # Overlaps but wrong type
    true_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
    ]
    pred_entities = [
        Entity(entity_type="PER", start=3, end=7),
        Entity(entity_type="LOC", start=10, end=15),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "PER", "start": 3, "end": 7, "precision": 0.0},
        {"entity_type": "LOC", "start": 10, "end": 15, "precision": 0.0},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 0.0},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 0.0},
    ]


def test_compute_precisions_recalls_for_type_doesnt_matter():
    # Overlaps but wrong type
    true_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
    ]
    pred_entities = [
        Entity(entity_type="PER", start=3, end=7),
        Entity(entity_type="LOC", start=10, end=15),
    ]
    # type DOES NOT matter
    label_to_int = {"LOC": 1, "PER": 1}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "PER", "start": 3, "end": 7, "precision": 1.0},
        {"entity_type": "LOC", "start": 10, "end": 15, "precision": 1.0},
    ]
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 1.0},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 1.0},
    ]


def test_compute_precisions_recalls_for_no_pred():
    true_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
    ]
    pred_entities: List = []
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == []
    assert recalls == [
        {"entity_type": "LOC", "start": 3, "end": 7, "recall": 0.0},
        {"entity_type": "PER", "start": 10, "end": 15, "recall": 0.0},
    ]


def test_compute_precisions_recalls_for_no_true():
    true_entities: List = []
    pred_entities = [
        Entity(entity_type="LOC", start=3, end=7),
        Entity(entity_type="PER", start=10, end=15),
    ]
    # type matters
    label_to_int = {"LOC": 1, "PER": 2}

    precisions = compute_entity_precisions_for_prediction(
        50, true_entities, pred_entities, label_to_int
    )
    recalls = compute_entity_recalls_for_ground_truth(
        50, true_entities, pred_entities, label_to_int
    )
    assert precisions == [
        {"entity_type": "LOC", "start": 3, "end": 7, "precision": 0.0},
        {"entity_type": "PER", "start": 10, "end": 15, "precision": 0.0},
    ]
    assert recalls == []


def test_compute_pii_detection_f1_for_no_recall_threshold_f1_is_one():
    precisions = [1.0, 1.0]
    recalls = [1.0, 1.0]
    actual = compute_pii_detection_f1(precisions, recalls)
    assert actual == 1.0


def test_compute_pii_detection_f1_for_no_recall_threshold_f1_is_zero():
    precisions = [0.0, 0.0]
    recalls = [0.0, 0.0]
    actual = compute_pii_detection_f1(precisions, recalls)
    assert actual == 0.0


def test_compute_pii_detection_f1_for_no_recall_threshold():
    precisions = [0.4, 0.8]
    recalls = [0.2, 0.7]
    actual = compute_pii_detection_f1(precisions, recalls)
    assert_almost_equal(actual, 0.5142857)


def test_compute_pii_detection_f1_with_recall_threshold():
    precisions = [0.4, 0.8, 0.9]
    recalls = [0.2, 0.51, 0.7]
    actual = compute_pii_detection_f1(precisions, recalls, recall_threshold=0.5)
    assert_almost_equal(actual, 0.716279)
