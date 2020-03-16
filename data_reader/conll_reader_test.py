from .conll_reader import sent2labels, sent2tokens
from pytest import fixture


@fixture
def sent():
    return [
        ("SOCCER", "NN", "O"),
        ("-", ":", "O"),
        ("JAPAN", "NNP", "I-LOC"),
        ("GET", "VB", "O"),
    ]


def test_sent2tokens(sent):
    actual = sent2tokens(sent)
    assert actual == ["SOCCER", "-", "JAPAN", "GET"]


def test_sent2labels(sent):
    acutal = sent2labels(sent)
    assert acutal == ["O", "O", "I-LOC", "O"]
