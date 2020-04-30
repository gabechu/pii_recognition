from typing import List
from unittest.mock import Mock, patch

from pytest import fixture

from .conll_reader import ConllReader, _sent2labels, _sent2tokens, detokeniser_registry


@fixture
def sent():
    return [
        ("SOCCER", "NN", "O"),
        ("-", ":", "O"),
        ("JAPAN", "NNP", "I-LOC"),
        ("GET", "VB", "O"),
    ]


def get_mock_ConllCorpusReader():
    mock = Mock()
    mock.return_value.iob_sents.return_value = [
        [
            ("SOCCER", "NN", "O"),
            ("-", ":", "O"),
            ("JAPAN", "NNP", "I-LOC"),
            ("GET", "VB", "O"),
        ],
        [],
        [("Nadim", "NNP", "I-PER"), ("Ladki", "NNP", "I-PER")],
        [],
    ]
    return mock


def get_mock_detokeniser():
    def simple_detokeniser(tokens: List[str]) -> str:
        return " ".join(tokens)

    mock = Mock()
    mock.return_value.detokenise = simple_detokeniser

    return mock


@patch(
    "pii_recognition.data_readers.conll_reader.ConllCorpusReader",
    new=get_mock_ConllCorpusReader(),
)
@patch.object(
    target=detokeniser_registry,
    attribute="create_instance",
    new_callable=get_mock_detokeniser,
)
def test_get_conll_eval_data(mock_detokeniser):
    reader = ConllReader(detokeniser_setup={"name": "fake_reader"})
    sents, labels = reader.get_test_data(file_path="fake_path")
    mock_detokeniser.assert_called_once_with(config=None, name="fake_reader")
    assert sents == ["SOCCER - JAPAN GET", "Nadim Ladki"]
    assert labels == [["O", "O", "I-LOC", "O"], ["I-PER", "I-PER"]]


def test_sent2tokens(sent):
    actual = _sent2tokens(sent)
    assert actual == ["SOCCER", "-", "JAPAN", "GET"]


def test_sent2labels(sent):
    acutal = _sent2labels(sent)
    assert acutal == ["O", "O", "I-LOC", "O"]
