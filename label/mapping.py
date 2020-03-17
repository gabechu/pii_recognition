from typing import Dict, List


def map_labels(type_A_labels: List[str], A2B_mapping: Dict[str, str]) -> List[str]:
    """
    Convert between different label conventions. For example, in one convention
    entities are defined as [PER, LOC], while in another entities can be defined as
    [PERSON, LOCATION].
    """
    A_set = A2B_mapping.keys()

    type_B_labels = []
    for label in type_A_labels:
        if label in A_set:
            type_B_labels.append(A2B_mapping[label])
        else:
            type_B_labels.append(label)
    return type_B_labels


def mask_labels(
    input_labels: List[str], non_mask_labels: List[str], mask_value: str = "O"
) -> List[str]:
    results = []
    for lab in input_labels:
        if lab in non_mask_labels:
            results.append(lab)
        else:
            results.append(mask_value)
    return results
