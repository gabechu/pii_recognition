from itertools import chain
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from tokenizers.nltk_tokenizer import word_tokenizer
from recognisers.entity_recogniser import Rec_co


def get_predicted_entities(
    text: str,
    entities: List,
    recogniser: Rec_co,
    mapping: Optional[Dict] = None,
    default_label: str = "O",
) -> List:
    tokens = word_tokenizer(text)
    recognised_entities = recogniser.analyze(text, entities)

    entity_labels = [default_label] * len(tokens)
    for i in range(len(tokens)):
        current_token = tokens[i]
        for entity in recognised_entities:
            if is_substring(
                (current_token.start, current_token.end), (entity.start, entity.end)
            ):
                entity_labels[i] = (
                    mapping[entity.entity_type] if mapping else entity.entity_type
                )
                break
    return entity_labels


def is_substring(source_start_end: Tuple, target_start_end: Tuple) -> bool:
    if (
        source_start_end[0] >= target_start_end[0]
        and source_start_end[1] <= target_start_end[1]
    ):
        return True
    else:
        return False


def bio_classification_report(y_true: List, y_pred: List) -> str:
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {"O"}
    taglist = sorted(tagset, key=lambda tag: tag.split("-", 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in taglist],
        target_names=taglist,
    )
