from mock import patch
from pii_recognition.data_readers.data import Data, DataItem
from pii_recognition.evaluation.character_level_evaluation import (
    TextScore,
    EntityPrecision,
    EntityRecall,
)
from pii_recognition.labels.schema import Entity
from pytest import fixture

from .pii_validation_pipeline import (
    identify_pii_entities,
    calculate_precisions_and_recalls,
)


@fixture
def data():
    items = [
        DataItem(
            "It's like that since 12/17/1967", true_labels=[Entity("BIRTHDAY", 21, 31)]
        ),
        DataItem(
            "The address of Balefire Global is Valadouro 3, Ubide 48145",
            true_labels=[Entity("ORGANIZATION", 15, 30), Entity("LOCATION", 34, 58)],
        ),
    ]

    return Data(
        items,
        supported_entities={"BIRTHDAY", "ORGANIZATION", "LOCATION"},
        is_io_schema=False,
    )


@patch("pii_recognition.pipelines.pii_validation_pipeline.recogniser_registry")
def test_identify_pii_entities(mock_registry, data):
    mock_registry.create_instance.return_value.analyse.return_value = [
        Entity("test", 0, 4)
    ]

    data = identify_pii_entities(
        data,
        "test_recogniser",
        {"supported_entities": ["test"], "supported_languages": ["test"]},
    )

    assert [item.text for item in data.items] == [
        "It's like that since 12/17/1967",
        "The address of Balefire Global is Valadouro 3, Ubide 48145",
    ]
    assert [item.true_labels for item in data.items] == [
        [Entity("BIRTHDAY", 21, 31)],
        [Entity("ORGANIZATION", 15, 30), Entity("LOCATION", 34, 58)],
    ]
    assert [item.pred_labels for item in data.items] == [
        [Entity("test", 0, 4)],
        [Entity("test", 0, 4)],
    ]


def test_calculate_precisions_and_recalls_with_empty_predictions(data):
    grouped_targeted_labels = [{"BIRTHDAY"}, {"ORGANIZATION"}, {"LOCATION"}]

    actual = calculate_precisions_and_recalls(data, grouped_targeted_labels)
    assert len(actual) == 2
    assert actual[0] == TextScore(
        precisions=[], recalls=[EntityRecall(Entity("BIRTHDAY", 21, 31), 0.0)]
    )
    assert actual[1] == TextScore(
        precisions=[],
        recalls=[
            EntityRecall(Entity("ORGANIZATION", 15, 30), 0.0),
            EntityRecall(Entity("LOCATION", 34, 58), 0.0),
        ],
    )


def test_calculate_precisions_and_recalls_with_predictions(data):
    data.items[0].pred_labels = [Entity("BIRTHDAY", 0, 10)]
    data.items[1].pred_labels = [
        Entity("ORGANIZATION", 20, 30),
        Entity("LOCATION", 30, 46),
    ]
    grouped_targeted_labels = [{"BIRTHDAY"}, {"ORGANIZATION"}, {"LOCATION"}]

    actual = calculate_precisions_and_recalls(data, grouped_targeted_labels)
    assert len(actual) == 2
    assert actual[0] == TextScore(
        precisions=[EntityPrecision(Entity("BIRTHDAY", 0, 10), 0.0)],
        recalls=[EntityRecall(Entity("BIRTHDAY", 21, 31), 0.0)],
    )
    assert actual[1] == TextScore(
        precisions=[
            EntityPrecision(Entity("ORGANIZATION", 20, 30), 1.0),
            EntityPrecision(Entity("LOCATION", 30, 46), 0.75),
        ],
        recalls=[
            EntityRecall(Entity("ORGANIZATION", 15, 30), 2 / 3),
            EntityRecall(Entity("LOCATION", 34, 58), 0.5),
        ],
    )


def test_calculate_precisions_and_recalls_with_nontargeted_labels(data):
    grouped_targeted_labels = [{"ORGANIZATION"}, {"LOCATION"}]
    nontargeted_labels = {"BIRTHDAY", "DATE"}

    actual = calculate_precisions_and_recalls(
        data, grouped_targeted_labels, nontargeted_labels
    )
    assert len(actual) == 2
    assert actual[0] == TextScore(precisions=[], recalls=[],)
    assert actual[1] == TextScore(
        precisions=[],
        recalls=[
            EntityRecall(Entity("ORGANIZATION", 15, 30), 0.0),
            EntityRecall(Entity("LOCATION", 34, 58), 0.0),
        ],
    )
