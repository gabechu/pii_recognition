from typing import List
from unittest.mock import Mock, patch

from pytest import fixture, raises

from .conll_reader import ConllReader, _sent2labels, _sent2tokens


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


def get_mock_non_io_ConllCorpusReader():
    mock = Mock()
    mock.return_value.iob_sents.return_value = [
        [
            ("SOCCER", "NN", "O"),
            ("-", ":", "O"),
            ("JAPAN", "NNP", "I-LOC"),
            ("GET", "VB", "O"),
        ],
        [],
        [("Nadim", "NNP", "B-PER"), ("Ladki", "NNP", "I-PER")],
        [],
    ]
    return mock


@fixture
def mock_detokeniser():
    def simple_detokeniser(tokens: List[str]) -> str:
        return " ".join(tokens)

    mock = Mock()
    mock.detokenise = simple_detokeniser

    return mock


def test_get_conll_eval_data(mock_detokeniser):
    # test 1: succeed
    with patch(
        "pii_recognition.data_readers.conll_reader.ConllCorpusReader",
        new=get_mock_ConllCorpusReader(),
    ):
        reader = ConllReader(detokeniser=mock_detokeniser)
        data = reader.get_test_data(
            file_path="fake_path", supported_entities=["I-LOC", "I-PER"]
        )
        assert data.sentences == ["SOCCER - JAPAN GET", "Nadim Ladki"]
        assert data.labels == [["O", "O", "I-LOC", "O"], ["I-PER", "I-PER"]]
        assert data.supported_entities == ["I-LOC", "I-PER"]
        assert data.is_io_schema is True

    # test 2: using non-io schema
    with patch(
        "pii_recognition.data_readers.conll_reader.ConllCorpusReader",
        new=get_mock_non_io_ConllCorpusReader(),
    ):
        reader = ConllReader(detokeniser=mock_detokeniser)
        data = reader.get_test_data(
            file_path="fake_path",
            supported_entities=["I-LOC", "B-PER", "I-PER"],
            is_io_schema=False,
        )
        assert data.sentences == ["SOCCER - JAPAN GET", "Nadim Ladki"]
        assert data.labels == [["O", "O", "I-LOC", "O"], ["B-PER", "I-PER"]]
        assert data.supported_entities == ["I-LOC", "B-PER", "I-PER"]
        assert data.is_io_schema is False

    # test 3: contains unsupported entities
    with patch(
        "pii_recognition.data_readers.conll_reader.ConllCorpusReader",
        new=get_mock_ConllCorpusReader(),
    ):
        reader = ConllReader(detokeniser=mock_detokeniser)
        with raises(ValueError) as err:
            reader.get_test_data(
                file_path="fake_path", supported_entities=["I-LOC"], is_io_schema=False,
            )
        assert str(err.value) == (
            "Found unsupported entity {'I-PER'} in data. "
            "You may need to update your supported entity list."
        )


def test_sent2tokens(sent):
    actual = _sent2tokens(sent)
    assert actual == ["SOCCER", "-", "JAPAN", "GET"]


def test_sent2labels(sent):
    acutal = _sent2labels(sent)
    assert acutal == ["O", "O", "I-LOC", "O"]
