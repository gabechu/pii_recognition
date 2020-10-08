import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

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


def label_encoder(
    text_length: int, entities: List[Entity], label_to_int: Dict[Any, int],
) -> List[int]:
    """Encode entity labels into integers.

    Encode a text at character level according to its entity labels as well as a mapping
    defined by `label_to_int`. Note multi-tagging is not supported. One entity can have
    only one label tag. Non-entities are encoded to 0 and any entities you are not
    interested could get encoded to 0.

    Args:
        text_length: length of a text.
        entities: entities in a text identified by entity_type, start, end.
        label_to_int: a mapping between entity labels and integers.

    Returns:
        Integer code of the text.
    """
    # TODO: update tests
    code = [0] * text_length
    removed_labels = [key for key, value in label_to_int.items() if value == 0]
    logging.info(
        f"Removing the following entity types from evaluation: {removed_labels}."
    )

    for span in entities:
        label_name = span.entity_type
        if label_to_int[label_name] in removed_labels:
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
    label_mapping: Dict,
) -> List[EntityPrecision]:
    """Compute precision for every entity in prediction."""
    true_code: List[int] = label_encoder(text_length, true_entities, label_mapping)

    precisions = []
    for pred_entity in pred_entities:
        int_label: int = label_mapping[pred_entity.entity_type]
        # 0 labels are ignored from calculation
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
        # 0 labels are ignored from calculation
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
