from unittest.mock import Mock, patch

from pytest import fixture

from pii_recognition.tokenisation.detokenisers import SpaceJoinDetokeniser

from .conll_reader import get_conll_eval_data, sent2labels, sent2tokens


# temporary fix and will update
@fixture
def detokeniser():
    return SpaceJoinDetokeniser().detokenise


@fixture
def sent():
    return [
        ("SOCCER", "NN", "O"),
        ("-", ":", "O"),
        ("JAPAN", "NNP", "I-LOC"),
        ("GET", "VB", "O"),
    ]


def mock_ConllCorpusReader():
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


@patch(
    "pii_recognition.data_readers.conll_reader.ConllCorpusReader",
    new=mock_ConllCorpusReader(),
)
def test_get_conll_eval_data(detokeniser):
    sents, labels = get_conll_eval_data(file_path="fake_path", detokenizer=detokeniser)
    assert sents == ["SOCCER - JAPAN GET", "Nadim Ladki"]
    assert labels == [["O", "O", "I-LOC", "O"], ["I-PER", "I-PER"]]


def test_sent2tokens(sent):
    actual = sent2tokens(sent)
    assert actual == ["SOCCER", "-", "JAPAN", "GET"]


def test_sent2labels(sent):
    acutal = sent2labels(sent)
    assert acutal == ["O", "O", "I-LOC", "O"]
