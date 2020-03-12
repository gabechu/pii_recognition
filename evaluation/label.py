from typing import Dict, List, NamedTuple


class Label(NamedTuple):
    annotated: str
    predicted: str


def map_labels(
    schema_A_labels: List[str], A2B_mapping: Dict[str, str]
) -> List[str]:
    """Convert between different labeling schema."""
    A_label_set = A2B_mapping.keys()

    B_labels = []
    for label in schema_A_labels:
        if label in A_label_set:
            B_labels.append(A2B_mapping[label])
        else:
            B_labels.append(label)
    return B_labels
