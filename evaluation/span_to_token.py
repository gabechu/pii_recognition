from typing import Dict, List, Tuple

from recognisers.recogniser_result import RecogniserResult
from tokenizers.token import Token


def is_substring(
    source_start_end: Tuple[int, int], target_start_end: Tuple[int, int]
) -> bool:
    """
    Whether a string is a substring of another. The string is annotated by position
    of the start and end characters.
    """
    if (
        source_start_end[0] >= target_start_end[0]
        and source_start_end[1] <= target_start_end[1]
    ):
        return True
    else:
        return False


def span_labels_to_token_labels(
    tokens: List[Token], recognised_entities: List[RecogniserResult]
) -> List[str]:
    """
    A conversion that breaks entity labeled by spans to tokens.

    Args:
        tokens: Text into tokens.
        recognised_entities: Model predicted entities.

    Returns:
        Token based entity labels.
    """
    token_entity_labels = ["O"] * len(tokens)
    for i in range(len(tokens)):
        current_token = tokens[i]
        for entity in recognised_entities:
            if is_substring(
                (current_token.start, current_token.end), (entity.start, entity.end)
            ):
                token_entity_labels[i] = entity.entity_type
                break
    return token_entity_labels
