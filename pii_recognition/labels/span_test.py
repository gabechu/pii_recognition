import pytest

from pii_recognition.tokenisation.token_schema import Token

from .schema import Entity, TokenLabel
from .span import is_substring, span_labels_to_token_labels, token_labels_to_span_labels


def test_is_substring():
    assert is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(4, 8))
    assert is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(5, 8))
    assert is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(4, 9))

    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(6, 8))
    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(6, 9))
    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(5, 7))
    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(4, 7))


def test_span_labels_to_token_labels():
    # reference sentence: "This is Bob Smith from Melbourne."
    span_labels = [
        Entity("PER", 8, 17),
        Entity("LOC", 23, 32),
    ]
    tokens = [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("Bob", 8, 11),
        Token("Smith", 12, 17),
        Token("from", 18, 22),
        Token("Melbourne", 23, 32),
        Token(".", 32, 33),
    ]
    actual = span_labels_to_token_labels(span_labels, tokens)
    assert [x.entity_type for x in actual] == ["O", "O", "PER", "PER", "O", "LOC", "O"]

    actual = span_labels_to_token_labels(span_labels, tokens, keep_o_label=False)
    assert [x.entity_type for x in actual] == ["PER", "PER", "LOC"]


def test_token_labels_to_span_labels():
    #
    token_labels = [TokenLabel("PER", 0, 4)]
    actual = token_labels_to_span_labels(token_labels)
    assert actual == [Entity("PER", 0, 4)]

    # text: Luke
    token_labels = [
        TokenLabel("PER", 0, 4),
        TokenLabel("PER", 5, 14),
    ]
    actual = token_labels_to_span_labels(token_labels)
    assert actual == [Entity("PER", 0, 14)]

    # text: Luke Skywalker
    token_labels = [
        TokenLabel("PER", 0, 4),
        TokenLabel("PER", 5, 14),
        TokenLabel("O", 14, 15),
    ]
    actual = token_labels_to_span_labels(token_labels)
    assert actual == [Entity("PER", 0, 14), Entity("O", 14, 15)]

    # text: Luke-Skywalker
    token_labels = [TokenLabel("PER", 0, 5), TokenLabel("PER", 5, 14)]
    actual = token_labels_to_span_labels(token_labels)
    assert actual == [Entity("PER", 0, 14)]

    # text: one day, Luke Skywalker and Wedge Antilles recover a message
    token_labels = [
        TokenLabel("O", 0, 3),
        TokenLabel("O", 4, 7),
        TokenLabel("O", 7, 8),
        TokenLabel("PER", 9, 13),
        TokenLabel("PER", 14, 23),
        TokenLabel("O", 24, 27),
        TokenLabel("PER", 28, 33),
        TokenLabel("PER", 34, 42),
        TokenLabel("O", 43, 50),
        TokenLabel("O", 51, 52),
        TokenLabel("O", 53, 60),
    ]
    actual = token_labels_to_span_labels(token_labels)
    assert actual == [
        Entity("O", 0, 8),
        Entity("PER", 9, 23),
        Entity("O", 24, 27),
        Entity("PER", 28, 42),
        Entity("O", 43, 60),
    ]

    # test failure
    token_labels = [TokenLabel("O", 4, 7), TokenLabel("O", 0, 3), TokenLabel("O", 7, 8)]
    with pytest.raises(AssertionError) as err:
        token_labels_to_span_labels(token_labels)
    assert str(err.value) == "token_labels are not in ascending order"
