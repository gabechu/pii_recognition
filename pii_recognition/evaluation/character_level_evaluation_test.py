from typing import List

import pytest
from numpy.testing import assert_almost_equal
from pii_recognition.evaluation.character_level_evaluation import (
    build_label_mapping,
    compute_entity_precisions_for_prediction,
    compute_entity_recalls_for_ground_truth,
    compute_pii_detection_f1,
    label_encoder,
    EntityRecall,
    EntityPrecision,
)
from pii_recognition.labels.schema import Entity


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
        EntityPrecision(Entity(entity_type="LOC", start=3, end=7), 1.0),
        EntityPrecision(Entity(entity_type="PER", start=10, end=15), 1.0),
        EntityPrecision(Entity(entity_type="LOC", start=23, end=32), 1.0),
        EntityPrecision(Entity(entity_type="PER", start=37, end=48), 1.0),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 1.0),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 1.0),
        EntityRecall(Entity(entity_type="LOC", start=23, end=32), 1.0),
        EntityRecall(Entity(entity_type="PER", start=37, end=48), 1.0),
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
        EntityPrecision(Entity(entity_type="LOC", start=4, end=7), 1.0),
        EntityPrecision(Entity(entity_type="PER", start=13, end=15), 1.0),
        EntityPrecision(Entity(entity_type="LOC", start=25, end=30), 1.0),
        EntityPrecision(Entity(entity_type="PER", start=40, end=46), 1.0),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 0.75),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 0.4),
        EntityRecall(Entity(entity_type="LOC", start=23, end=32), 5 / 9),
        EntityRecall(Entity(entity_type="PER", start=37, end=48), 6 / 11),
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
        EntityPrecision(Entity(entity_type="LOC", start=3, end=7), 0.75),
        EntityPrecision(Entity(entity_type="PER", start=10, end=15), 0.4),
        EntityPrecision(Entity(entity_type="LOC", start=23, end=32), 5 / 9),
        EntityPrecision(Entity(entity_type="PER", start=37, end=48), 6 / 11),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=4, end=7), 1.0),
        EntityRecall(Entity(entity_type="PER", start=13, end=15), 1.0),
        EntityRecall(Entity(entity_type="LOC", start=25, end=30), 1.0),
        EntityRecall(Entity(entity_type="PER", start=40, end=46), 1.0),
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
        EntityPrecision(Entity(entity_type="LOC", start=1, end=4), 1 / 3),
        EntityPrecision(Entity(entity_type="PER", start=13, end=18), 0.4),
        EntityPrecision(Entity(entity_type="LOC", start=28, end=35), 4 / 7),
        EntityPrecision(Entity(entity_type="PER", start=45, end=49), 0.75),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 0.25),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 0.4),
        EntityRecall(Entity(entity_type="LOC", start=23, end=32), 4 / 9),
        EntityRecall(Entity(entity_type="PER", start=37, end=48), 3 / 11),
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
        EntityPrecision(Entity(entity_type="LOC", start=15, end=20), 0.0),
        EntityPrecision(Entity(entity_type="PER", start=33, end=35), 0.0),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 0.0),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 0.0),
        EntityRecall(Entity(entity_type="LOC", start=23, end=32), 0.0),
        EntityRecall(Entity(entity_type="PER", start=37, end=48), 0.0),
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
        EntityPrecision(Entity(entity_type="LOC", start=3, end=20), 4 / 17),
        EntityPrecision(Entity(entity_type="PER", start=28, end=43), 0.4),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 1.0),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 0.0),
        EntityRecall(Entity(entity_type="LOC", start=23, end=32), 0.0),
        EntityRecall(Entity(entity_type="PER", start=37, end=48), 6 / 11),
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
        EntityPrecision(Entity(entity_type="LOC", start=3, end=7), 1.0),
        EntityPrecision(Entity(entity_type="PER", start=10, end=15), 0.0),
        EntityPrecision(Entity(entity_type="LOC", start=23, end=32), 0.0),
        EntityPrecision(Entity(entity_type="PER", start=37, end=48), 6 / 11),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=20), 4 / 17),
        EntityRecall(Entity(entity_type="PER", start=28, end=43), 0.4),
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
        EntityPrecision(Entity(entity_type="PER", start=3, end=7), 0.0),
        EntityPrecision(Entity(entity_type="LOC", start=10, end=15), 0.0),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 0.0),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 0.0),
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
        EntityPrecision(Entity(entity_type="PER", start=3, end=7), 1.0),
        EntityPrecision(Entity(entity_type="LOC", start=10, end=15), 1.0),
    ]
    assert recalls == [
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 1.0),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 1.0),
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
        EntityRecall(Entity(entity_type="LOC", start=3, end=7), 0.0),
        EntityRecall(Entity(entity_type="PER", start=10, end=15), 0.0),
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
        EntityPrecision(Entity(entity_type="LOC", start=3, end=7), 0.0),
        EntityPrecision(Entity(entity_type="PER", start=10, end=15), 0.0),
    ]
    assert recalls == []


def test_compute_entity_precisions_for_prediction_no_true_entities():
    actual = compute_entity_precisions_for_prediction(
        50, [], [Entity("PER", 5, 10), Entity("LOC", 15, 25)], {"PER": 1, "LOC": 2}
    )
    assert actual == [
        EntityPrecision(Entity("PER", 5, 10), 0.0),
        EntityPrecision(Entity("LOC", 15, 25), 0.0),
    ]


def test_compute_entity_precisions_for_prediction_no_pred_entities():
    actual = compute_entity_precisions_for_prediction(
        50, [Entity("PER", 5, 10), Entity("LOC", 15, 25)], [], {"PER": 1, "LOC": 2}
    )
    assert actual == []


def test_compute_entity_precisions_for_prediction_no_true_no_pred_entities():
    actual = compute_entity_precisions_for_prediction(50, [], [], {"PER": 1, "LOC": 2})
    assert actual == []


def test_compute_entity_recalls_for_ground_truth_no_true_entities():
    actual = compute_entity_recalls_for_ground_truth(
        50, [], [Entity("PER", 5, 10), Entity("LOC", 15, 25)], {"PER": 1, "LOC": 2}
    )
    assert actual == []


def test_compute_entity_recalls_for_ground_truth_no_pred_entities():
    actual = compute_entity_recalls_for_ground_truth(
        50, [Entity("PER", 5, 10), Entity("LOC", 15, 25)], [], {"PER": 1, "LOC": 2}
    )
    assert actual == [
        EntityRecall(Entity("PER", 5, 10), 0.0),
        EntityRecall(Entity("LOC", 15, 25), 0.0),
    ]


def test_compute_entity_recalls_for_ground_truth_no_true_pred_entities():
    actual = compute_entity_recalls_for_ground_truth(50, [], [], {"PER": 1, "LOC": 2})
    assert actual == []


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


def test_compute_pii_detection_f1_for_invalid_threshold():
    precisions = recalls = [0.0]
    with pytest.raises(ValueError) as err:
        compute_pii_detection_f1(precisions, recalls, recall_threshold=2.0)
    assert str(err.value) == (
        "Invalid threshold! Recall threshold must between 0 and 1 but got 2.0"
    )


def test_compute_pii_detection_f1_for_empty_precisions():
    with pytest.raises(ValueError) as err:
        compute_pii_detection_f1([], [0.0], recall_threshold=0.5)
    assert str(err.value) == "You are passing empty precisions list!"


def test_compute_pii_detection_f1_for_empty_recalls():
    with pytest.raises(ValueError) as err:
        compute_pii_detection_f1([0.0], [], recall_threshold=0.5)
    assert str(err.value) == "You are passing empty recalls list!"


def test_compute_pii_detection_f1_for_empty_precisions_recalls():
    with pytest.raises(ValueError) as err:
        compute_pii_detection_f1([], [], recall_threshold=0.5)
    assert str(err.value) == "You are passing empty precisions and recalls lists!"


def test_build_label_mapping_with_nontargeted_labels():
    grouped_targeted_labels = [{"PERSON", "PER"}, {"LOCATION"}, {"DATE"}]
    nontargeted_labels = {"NONTARGETED"}

    actual = build_label_mapping(grouped_targeted_labels, nontargeted_labels)
    assert actual == {"PERSON": 1, "PER": 1, "LOCATION": 2, "DATE": 3, "NONTARGETED": 0}


def test_build_label_mapping_without_nontargeted_labels():
    grouped_targeted_labels = [{"PERSON", "PER"}, {"LOCATION"}, {"DATE"}]

    actual = build_label_mapping(grouped_targeted_labels)
    assert actual == {"PERSON": 1, "PER": 1, "LOCATION": 2, "DATE": 3}
