from dataclasses import asdict
from typing import Any, Dict, List, Optional

from pii_recognition.evaluation.metrics import (
    compute_f_beta,
    compute_label_precision,
    compute_label_recall,
)
from pii_recognition.labels.schema import Entity


def label_encoder(
    text_length: int, entities: List[Entity], label_to_int: Dict[Any, int],
) -> List[int]:
    """Encode entity labels into integers.

    Encode a text at character level according to its entity labels as well as a mapping
    defined by `label_to_int`. Note multi-tagging is not supported. One entity can have
    only one label tag.

    Args:
        text_length: length of a text.
        entities: entities in a text identified by entity_type, start, end.
        label_to_int: a mapping between entity labels and integers.

    Returns:
        Integer code of the text.
    """
    if 0 in label_to_int.values():
        raise ValueError(
            "Value 0 is reserved! If a character does not belong to any entity, it "
            "would be assigned with 0."
        )
    code = [0] * text_length

    for span in entities:
        s = span.start
        e = span.end
        if e > text_length:
            raise ValueError(
                f"Entity span index is out of range: text length is "
                f"{text_length} but got span index {e}."
            )
        label_name = span.entity_type
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
) -> List[Dict]:
    """Compute precision for every entity in prediction."""
    true_code: List[int] = label_encoder(text_length, true_entities, label_mapping)

    precisions = []
    for pred_entity in pred_entities:
        pred_entity_code: List[int] = label_encoder(
            text_length, [pred_entity], label_mapping
        )
        int_label: int = label_mapping[pred_entity.entity_type]
        score = compute_label_precision(true_code, pred_entity_code, int_label)

        entity_dict = asdict(pred_entity)
        entity_dict.update({"precision": score})
        precisions.append(entity_dict)

    return precisions


def compute_entity_recalls_for_ground_truth(
    text_length: int,
    true_entities: List[Entity],
    pred_entities: List[Entity],
    label_mapping: Dict,
) -> List[Dict]:
    """Compute recall for every entity in ground truth."""
    pred_code: List[int] = label_encoder(text_length, pred_entities, label_mapping)

    recalls = []
    for true_entity in true_entities:
        true_entity_code: List[int] = label_encoder(
            text_length, [true_entity], label_mapping
        )
        int_label: int = label_mapping[true_entity.entity_type]
        score = compute_label_recall(true_entity_code, pred_code, int_label)

        entity_dict = asdict(true_entity)
        entity_dict.update({"recall": score})
        recalls.append(entity_dict)

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
