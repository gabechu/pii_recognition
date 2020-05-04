from dataclasses import asdict
from typing import Dict, List, Tuple

from pii_recognition.tokenisation.token_schema import Token
from pii_recognition.utils import is_ascending

from .schema import SpanLabel, TokenLabel


def is_substring(
    segment_A_start_end: Tuple[int, int], segment_B_start_end: Tuple[int, int]
) -> bool:
    """
    Whether segment A is a substring of segment B, where segment is identified by
    indices of the start and end characters.
    """
    if (
        segment_A_start_end[0] >= segment_B_start_end[0]
        and segment_A_start_end[1] <= segment_B_start_end[1]
    ):
        return True
    else:
        return False


def span_labels_to_token_labels(
    span_labels: List[SpanLabel], tokens: List[Token], keep_o_label: bool = True
) -> List[TokenLabel]:
    """
    A conversion that breaks entity labeled by spans to tokens.

    Args:
        tokens: Text into tokens.
        recognised_entities: Model predicted entities.

    Returns:
        Token based entity labels, e.g., ["O", "O", "LOC", "O"].
    """
    labels = ["O"] * len(tokens)  # default is O, no chunck label

    for i in range(len(tokens)):
        current_token = tokens[i]
        for label in span_labels:
            if is_substring(
                (current_token.start, current_token.end), (label.start, label.end)
            ):
                labels[i] = label.entity_type
                break

    if keep_o_label:
        return [
            TokenLabel(entity_type=labels[i], start=tokens[i].start, end=tokens[i].end)
            for i in range(len(tokens))
        ]
    else:
        return [
            TokenLabel(entity_type=labels[i], start=tokens[i].start, end=tokens[i].end)
            for i in range(len(tokens))
            if labels[i] != "O"
        ]


def token_labels_to_span_labels(token_labels: List[TokenLabel]) -> List[SpanLabel]:
    # order matters
    assert is_ascending(
        [label.start for label in token_labels]
    ), "token_labels are not in ascending order"

    span_labels = []
    prior_span: Dict = asdict(token_labels[0])

    for i in range(1, len(token_labels)):
        current_token_label = token_labels[i]
        if current_token_label.entity_type == prior_span["entity_type"]:
            prior_span["end"] = current_token_label.end
        else:
            span_labels.append(SpanLabel(**prior_span))
            prior_span = asdict(current_token_label)

    span_labels.append(SpanLabel(**prior_span))

    return span_labels
