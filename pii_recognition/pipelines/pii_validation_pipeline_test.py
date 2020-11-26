import os
from tempfile import TemporaryDirectory

from mock import patch
from numpy.testing import assert_array_almost_equal
from pii_recognition.data_readers.data import Data, DataItem
from pii_recognition.evaluation.character_level_evaluation import (
    EntityPrecision,
    EntityRecall,
    TextScore,
)
from pii_recognition.labels.schema import Entity
from pii_recognition.utils import load_json_file
from pytest import fixture

from .pii_validation_pipeline import (
    calculate_precisions_and_recalls,
    get_rollup_fscore_on_pii,
    get_rollup_metrics_on_types,
    identify_pii_entities,
    log_predictions_and_ground_truths,
    regroup_scores_on_types,
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


@fixture
def scores():
    scores = []
    scores.append(
        TextScore(
            text="It's like that since 9/23/1993",
            precisions=[EntityPrecision(Entity("BIRTHDAY", 0, 10), 0.0)],
            recalls=[EntityRecall(Entity("BIRTHDAY", 21, 31), 0.0)],
        )
    )
    scores.append(
        TextScore(
            text="The address of Balefire Global is Valadouro 3, Ubide 48145",
            precisions=[
                EntityPrecision(Entity("ORGANIZATION", 20, 30), 1.0),
                EntityPrecision(Entity("LOCATION", 30, 46), 0.75),
            ],
            recalls=[
                EntityRecall(Entity("ORGANIZATION", 15, 30), 2 / 3),
                EntityRecall(Entity("LOCATION", 34, 58), 0.5),
            ],
        )
    )

    return scores


@fixture
def complex_scores():
    scores = []

    # 1. test grouping i.e. DATE and BIRTHDAY
    scores.append(
        TextScore(
            text="It's like that since 12/17/1967",
            precisions=[EntityPrecision(Entity("DATE", 0, 10), 0.0)],
            recalls=[EntityRecall(Entity("BIRTHDAY", 21, 31), 0.0)],
        )
    )

    # 2. test removal of non-interested i.e. ORGANIZATION
    scores.append(
        TextScore(
            text="The address of Balefire Global is Valadouro 3, Ubide 48145",
            precisions=[
                EntityPrecision(Entity("ORGANIZATION", 20, 30), 1.0),
                EntityPrecision(Entity("LOCATION", 30, 46), 0.75),
            ],
            recalls=[
                EntityRecall(Entity("ORGANIZATION", 15, 30), 2 / 3),
                EntityRecall(Entity("LOCATION", 34, 58), 0.5),
            ],
        )
    )

    # 3. test multiple occurrences of a type i.e. LOCATION it occurs in
    # case 4 and 5 as well, this will test calculation on aggregated scores.
    scores.append(
        TextScore(
            text=(
                "Please update billing addrress with Slovenčeva 71, "
                "Dol pri Ljubljani 1262 for this card: 4539881821557738"
            ),
            precisions=[
                EntityPrecision(Entity("LOCATION", 26, 66), 0.75),
                EntityPrecision(Entity("CREDIT_CARD", 89, 105), 1.0),
            ],
            recalls=[
                EntityRecall(Entity("LOCATION", 36, 73), 30 / 37),
                EntityRecall(Entity("CREDIT_CARD", 89, 105), 1.0),
            ],
        )
    )

    # 4. test empty precisions
    scores.append(
        TextScore(
            text=(
                "I once lived in Árpád fejedelem útja 89., Bicske 2063. "
                "I now live in Sarandi 5156, 25 de Agosto 94002"
            ),
            precisions=[],
            recalls=[
                EntityRecall(Entity("LOCATION", 16, 53), 0.0),
                EntityRecall(Entity("LOCATION", 69, 101), 0.0),
            ],
        )
    )

    # 5. test empty recalls
    scores.append(
        TextScore(
            text="rory is from revelstone",
            precisions=[EntityPrecision(Entity("LOCATION", 13, 23), 0.0)],
            recalls=[],
        )
    )

    # 6. test empty precisions and recalls
    scores.append(
        TextScore(
            text="How can I request a new credit card pin ?", precisions=[], recalls=[]
        )
    )

    return scores


@patch("pii_recognition.pipelines.pii_validation_pipeline.recogniser_registry")
def test_identify_pii_entities(mock_registry, data):
    mock_registry.create_instance.return_value.analyse.return_value = [
        Entity("test", 0, 4)
    ]

    actual = identify_pii_entities(
        data,
        "test_recogniser",
        {"supported_entities": ["test"], "supported_languages": ["test"]},
    )

    assert [item.text for item in actual.items] == [
        "It's like that since 12/17/1967",
        "The address of Balefire Global is Valadouro 3, Ubide 48145",
    ]
    assert [item.true_labels for item in actual.items] == [
        [Entity("BIRTHDAY", 21, 31)],
        [Entity("ORGANIZATION", 15, 30), Entity("LOCATION", 34, 58)],
    ]
    assert [item.pred_labels for item in actual.items] == [
        [Entity("test", 0, 4)],
        [Entity("test", 0, 4)],
    ]


def test_calculate_precisions_and_recalls_with_empty_predictions(data):
    grouped_targeted_labels = [{"BIRTHDAY"}, {"ORGANIZATION"}, {"LOCATION"}]

    unwrapped = calculate_precisions_and_recalls(data, grouped_targeted_labels)
    actual = unwrapped["scores"]

    assert len(actual) == 2
    assert actual[0] == TextScore(
        text="It's like that since 12/17/1967",
        precisions=[],
        recalls=[EntityRecall(Entity("BIRTHDAY", 21, 31), 0.0)],
    )
    assert actual[1] == TextScore(
        text="The address of Balefire Global is Valadouro 3, Ubide 48145",
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

    unwrapped = calculate_precisions_and_recalls(data, grouped_targeted_labels)
    actual = unwrapped["scores"]

    assert len(actual) == 2
    assert actual[0] == TextScore(
        text="It's like that since 12/17/1967",
        precisions=[EntityPrecision(Entity("BIRTHDAY", 0, 10), 0.0)],
        recalls=[EntityRecall(Entity("BIRTHDAY", 21, 31), 0.0)],
    )
    assert actual[1] == TextScore(
        text="The address of Balefire Global is Valadouro 3, Ubide 48145",
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

    unwrapped = calculate_precisions_and_recalls(
        data, grouped_targeted_labels, nontargeted_labels
    )
    actual = unwrapped["scores"]

    assert len(actual) == 2
    assert actual[0] == TextScore(
        text="It's like that since 12/17/1967", precisions=[], recalls=[],
    )
    assert actual[1] == TextScore(
        text="The address of Balefire Global is Valadouro 3, Ubide 48145",
        precisions=[],
        recalls=[
            EntityRecall(Entity("ORGANIZATION", 15, 30), 0.0),
            EntityRecall(Entity("LOCATION", 34, 58), 0.0),
        ],
    )


def test_get_rollup_fscore_on_pii_no_threshold(scores):
    actual = get_rollup_fscore_on_pii(scores, fbeta=1, recall_threshold=None)
    assert actual == 0.35


def test_get_rollup_fscore_on_pii_threshold(scores):
    actual = get_rollup_fscore_on_pii(scores, fbeta=1, recall_threshold=0.4)
    assert actual == 0.4667


def test_get_rollup_metrics_on_types(complex_scores):
    actual = get_rollup_metrics_on_types(
        [{"BIRTHDAY", "DATE"}, {"LOCATION"}, {"CREDIT_CARD"}], complex_scores, 1.0
    )

    assert len(actual) == 3
    assert actual[frozenset({"BIRTHDAY", "DATE"})] == {
        "f1": 0.0,
        "ave-precision": 0.0,
        "ave-recall": 0.0,
    }
    assert actual[frozenset({"CREDIT_CARD"})] == {
        "f1": 1.0,
        "ave-precision": 1.0,
        "ave-recall": 1.0,
    }
    assert actual[frozenset({"LOCATION"})] == {
        "f1": 0.3959,
        "ave-precision": 0.5,
        "ave-recall": 0.3277,
    }


def test_get_rollup_metrics_on_types_empty_scores():
    actual = get_rollup_metrics_on_types([{"BIRTHDAY", "DATE"}], [], 1.0)

    assert actual == {
        frozenset({"BIRTHDAY", "DATE"}): {
            "f1": 1.0,
            "ave-precision": "undefined",
            "ave-recall": "undefined",
        }
    }


def test_log_mistakes(scores):
    with TemporaryDirectory() as tempdir:
        fake_path = os.path.join(tempdir, "fake_path")
        log_predictions_and_ground_truths(fake_path, scores)
        actual = load_json_file(fake_path)

    assert len(actual) == 2
    item_one = actual["It's like that since 9/23/1993"]
    assert item_one["predicted"] == {
        "It's like ": {"type": "BIRTHDAY", "score": 0.0, "start": 0}
    }
    assert item_one["ground_truth"] == {
        "9/23/1993": {"type": "BIRTHDAY", "score": 0.0, "start": 21},
    }
    item_two = actual["The address of Balefire Global is Valadouro 3, Ubide 48145"]
    assert item_two["predicted"] == {
        "ire Global": {"type": "ORGANIZATION", "score": 1.0, "start": 20},
        " is Valadouro 3,": {"type": "LOCATION", "score": 0.75, "start": 30},
    }
    assert item_two["ground_truth"] == {
        "Balefire Global": {"type": "ORGANIZATION", "score": 0.67, "start": 15},
        "Valadouro 3, Ubide 48145": {"type": "LOCATION", "score": 0.5, "start": 34},
    }


def test_regroup_scores_on_types(scores):
    actual = regroup_scores_on_types(
        [{"DATE", "BIRTHDAY"}, {"LOCATION"}, {"ORGANIZATION"}], scores
    )

    assert len(actual) == 3
    assert actual[frozenset({"DATE", "BIRTHDAY"})] == {
        "precisions": [0.0],
        "recalls": [0.0],
    }
    assert actual[frozenset({"LOCATION"})] == {"precisions": [0.75], "recalls": [0.5]}
    assert actual[frozenset({"ORGANIZATION"})]["precisions"] == [1.0]
    assert_array_almost_equal(
        actual[frozenset({"ORGANIZATION"})]["recalls"], [0.6666666666666666]
    )
