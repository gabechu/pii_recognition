from tokeniser.token import Token

from .label_schema import SpanLabel
from .span_to_token import is_substring, span_labels_to_token_labels


def test_is_substring():
    assert is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(4, 8))
    assert is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(5, 8))
    assert is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(4, 9))

    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(6, 8))
    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(6, 9))
    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(5, 7))
    assert not is_substring(segment_A_start_end=(5, 8), segment_B_start_end=(4, 7))


def test_span_labels_to_token_labels():
    span_labels = [SpanLabel("PER", 8, 11), SpanLabel("LOC", 17, 26)]
    tokens = [
        Token("This", 0, 4),
        Token("is", 5, 7),
        Token("Bob", 8, 11),
        Token("from", 12, 16),
        Token("Melbourne", 17, 26),
        Token(".", 26, 27),
    ]
    actual = span_labels_to_token_labels(span_labels, tokens)
    assert actual == ["O", "O", "PER", "O", "LOC", "O"]
