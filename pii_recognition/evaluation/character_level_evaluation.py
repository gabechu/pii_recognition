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
    entity_src: str = "predicted"


@dataclass
class EntityRecall:
    entity: Entity
    recall: float
    entity_src: str = "ground_truth"


@dataclass
class TextScore:
    precisions: List[EntityPrecision]
    recalls: List[EntityRecall]


def build_label_mapping(
    grouped_targeted_labels: List[Set[str]],
    nontargeted_labels: Optional[Set[str]] = None,
) -> Dict[str, int]:
    """Map an entity label to an integer.

    Create a dictionary for mapping entity labels. Targeted entity labels are mapped to
    a positive integer starting from 1 and labels within the same group are mapped to
    the same integer. Non-targeted labels are mapped to zero.

    Args:
        grouped_targeted_labels: entity labels we are interested that have been
            separated by groups.
        nontargeted_labels: entity labels we are not interested.

    Returns:
        A mapping dictionary.
    """
    mapping = {
        label: i + 1
        for i, label_group in enumerate(grouped_targeted_labels)
        for label in label_group
    }

    if nontargeted_labels:
        mapping.update({label: 0 for label in nontargeted_labels})

    return mapping


def label_encoder(
    text_length: int, entities: List[Entity], label_to_int: Dict[str, int],
) -> List[int]:
    """Encode entity labels into integers.

    Convert characters in a text to integers according to the entity labels provided
    and label_to_int dictionary where the dictionary gives a mapping between an entity
    label and an integer.

    Args:
        text_length: length of a text.
        entities: entities identified in a text.
        label_to_int: a dictionary that keys are entity labels and values are integers.

    Returns:
        Integer code of the text.
    """
    # note 0 means negative labels
    code = [0] * text_length

    for span in entities:
        label_name = span.entity_type
        try:
            label_code = label_to_int[label_name]
        except KeyError as err:
            raise Exception(f"Missing label {str(err)} in 'label_to_int' mapping.")

        if label_to_int[label_name] == 0:
            continue
        s = span.start
        e = span.end

        if e > text_length:
            raise ValueError(
                f"Entity span index is out of range: text length is "
                f"{text_length} but got span index {e}."
            )
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
        # note 0 means negative labels
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
        # note 0 means negative labels
        if int_label == 0:
            continue

        true_entity_code: List[int] = label_encoder(
            text_length, [true_entity], label_mapping
        )
        recall = compute_label_recall(true_entity_code, pred_code, int_label)
        recalls.append(EntityRecall(true_entity, recall))

    return recalls


def compute_pii_detection_fscore(
    precisions: List[float],
    recalls: List[float],
    recall_threshold: Optional[float] = None,
    beta: float = 1,
) -> float:
    """Evaluate performance of PII detection with F score.

    Rollup precisions and recalls to calculate F score on boundary detection for PII
    recognition. Recall thresholding is supported, if the system can recognise
    a certain portion of an entity beyond the threshold, we will consider it a
    success.

    Args:
        precisions: a list of entity precision values.
        recalls: a list of entity recall values.
        recall_threshold: a float between 0 and 1. Any recall value that is greater
            than or equals to the threshold would be rounded up to 1.

    Returns:
        F score.
    """
    if recall_threshold:
        if recall_threshold > 1.0 or recall_threshold < 0.0:
            raise ValueError(
                f"Invalid threshold! Recall threshold must between 0 and 1 "
                f"but got {recall_threshold}"
            )

    if not precisions and not recalls:
        # empty precisions and recalls mean that
        # there is no true entity and the system predicts nothing
        return 1.0
    if not precisions:
        # empty precisions mean that
        # there are true entities but the system predicts nothing
        return 0.0
    elif not recalls:
        # empty recalls mean that
        # there is no true entity but the system predicts something
        return 0.0

    if recall_threshold:
        recalls = [1.0 if item >= recall_threshold else item for item in recalls]

    ave_precision = sum(precisions) / len(precisions)
    ave_recall = sum(recalls) / len(recalls)
    return compute_f_beta(ave_precision, ave_recall, beta)
