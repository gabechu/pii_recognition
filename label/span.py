from typing import List, Tuple

from tokeniser.token import Token

from .label_schema import SpanLabel


def is_substring(
    segment_A_start_end: Tuple[int, int], segment_B_start_end: Tuple[int, int]
) -> bool:
    """
    Whether segment A is a substring of segment B, where segment is identified by
    position of the start and end characters.
    """
    if (
        segment_A_start_end[0] >= segment_B_start_end[0]
        and segment_A_start_end[1] <= segment_B_start_end[1]
    ):
        return True
    else:
        return False


def span_labels_to_token_labels(
    span_labels: List[SpanLabel], tokens: List[Token]
) -> List[str]:
    """
    A conversion that breaks entity labeled by spans to tokens.

    Args:
        tokens: Text into tokens.
        recognised_entities: Model predicted entities.

    Returns:
        Token based entity labels, e.g., ["O", "O", "LOC", "O"].
    """
    token_labels = ["O"] * len(tokens)

    for i in range(len(tokens)):
        current_token = tokens[i]
        for label in span_labels:
            if is_substring(
                (current_token.start, current_token.end), (label.start, label.end)
            ):
                token_labels[i] = label.entity_type
                break
    return token_labels
