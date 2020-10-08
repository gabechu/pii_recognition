from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from pii_recognition.evaluation.metrics import (
    compute_f_beta,
    compute_label_precision,
    compute_label_recall,
)
from pii_recognition.labels.schema import Entity


@dataclass
class EntityPrecision:
    entity: Entity
    precision: float


@dataclass
class EntityRecall:
    entity: Entity
    recall: float


@dataclass
class TicketScore:
    ticket_precisions: List[EntityPrecision]
    ticket_recalls: List[EntityRecall]


def build_label_mapping(
    grouped_targeted_labels: List[Set[str]], ignored_labels: Set[str]
) -> Dict[str, int]:
    """Map entity label to an integer.

    Create a dictionary for enity label mapping. Targeted labels would be mapped to a
    positive integer, if these labels are in the same group they will be mapped to the
    same integer. Non-targeted labels would map to zero.

    Args:
        grouped_labels: targeted entity labels. You can group these labels and each
            group would be mapped to an integer.
        ignored_labels: non-targeted entity labels. Those labels would be mapped to
            zero.

    Returns:
        A mapping dictionary.
    """
    return {
        **{
            label: i + 1
            for i, label_group in enumerate(grouped_targeted_labels)
            for label in label_group
        },
        **{label: 0 for label in ignored_labels},
    }


def label_encoder(
    text_length: int, entities: List[Entity], label_to_int: Dict[str, int]
) -> List[int]:
    """Encode entity labels into integers.

    Convert characters in a text to integers according to the entity labels provided
    and label_to_int dictionary.

    Args:
        text_length: length of a text.
        entities: entities identified in a text.
        label_to_int: a dictionary that keys are entity labels and values are integers.

    Returns:
        Integer code for text.
    """
    code = [0] * text_length

    for span in entities:
        label_name = span.entity_type
        # 0 refers to ignored labels
        if label_to_int[label_name] == 0:
            continue
        s = span.start
        e = span.end

        if e > text_length:
            raise ValueError(
                f"Entity span index is out of range: text length is "
                f"{text_length} but got span index {e}."
            )
        try:
            label_code = label_to_int[label_name]
        except KeyError as err:
            raise Exception(f"Missing label {str(err)} in 'label_to_int' mapping.")

        code[s:e] = [label_code] * (e - s)

    return code


def compute_entity_precisions_for_prediction(
    text_length: int,
    true_entities: List[Entity],
    pred_entities: List[Entity],
    label_mapping: Dict[str, int],
) -> List[EntityPrecision]:
    """Compute precision for every entity in prediction."""
    true_code: List[int] = label_encoder(text_length, true_entities, label_mapping)

    precisions = []
    for pred_entity in pred_entities:
        int_label: int = label_mapping[pred_entity.entity_type]
        # 0 refers to ignored labels
        if int_label == 0:
            continue
        pred_entity_code: List[int] = label_encoder(
            text_length, [pred_entity], label_mapping
        )
        precision = compute_label_precision(true_code, pred_entity_code, int_label)
        precisions.append(EntityPrecision(pred_entity, precision))

    return precisions


def compute_entity_recalls_for_ground_truth(
    text_length: int,
    true_entities: List[Entity],
    pred_entities: List[Entity],
    label_mapping: Dict,
) -> List[EntityRecall]:
    """Compute recall for every entity in ground truth."""
    pred_code: List[int] = label_encoder(text_length, pred_entities, label_mapping)

    recalls = []
    for true_entity in true_entities:
        int_label: int = label_mapping[true_entity.entity_type]
        # 0 refers to ignored labels
        if int_label == 0:
            continue
        true_entity_code: List[int] = label_encoder(
            text_length, [true_entity], label_mapping
        )
        recall = compute_label_recall(true_entity_code, pred_code, int_label)
        recalls.append(EntityRecall(true_entity, recall))

    return recalls


def compute_pii_detection_f1(
    precisions: List[float],
    recalls: List[float],
    recall_threshold: Optional[float] = None,
) -> float:
    """Evaluate performance of PII detection with F1.

    Rollup precisions and recalls to calculate F1 on boundary detection for PII
    recognition. Recall thresholding is supported, if the system can recognise
    a certain portion of an entity beyond the threshold, we will consider it a
    success.

    Args:
        precisions: a list of entity precision values.
        recalls: a list of entity recall values.
        recall_threshold: a float between 0 and 1. Any value greater than or equals
            to the threshold would be rounded up to 1.

    Returns:
        F1 score.
    """
    if recall_threshold:
        if recall_threshold > 1.0 or recall_threshold < 0.0:
            raise ValueError(
                f"Invalid threshold! Recall threshold must between 0 and 1 "
                f"but got {recall_threshold}"
            )

    if not precisions and not recalls:
        raise ValueError("You are passing empty precisions and recalls lists!")
    if not precisions:
        raise ValueError("You are passing empty precisions list!")
    elif not recalls:
        raise ValueError("You are passing empty recalls list!")

    if recall_threshold:
        recalls = [1.0 if item >= recall_threshold else item for item in recalls]

    ave_precision = sum(precisions) / len(precisions)
    ave_recall = sum(recalls) / len(recalls)
    return compute_f_beta(ave_precision, ave_recall)
