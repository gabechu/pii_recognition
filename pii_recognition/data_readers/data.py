from dataclasses import dataclass
from typing import Generic, List, Optional, Set, TypeVar

from pii_recognition.labels.schema import Entity

# Two kinds of entity labels
# 1. List[str] indicates every token has a label.
# 2. List[Entity] indicates every text-span has a label.
TEXT_LABELS = TypeVar("TEXT_LABELS", List[str], List[Entity])


@dataclass
class DataItem(Generic[TEXT_LABELS]):
    text: str
    true_label: TEXT_LABELS
    pred_label: Optional[TEXT_LABELS] = None


@dataclass
class Data:
    items: List[DataItem]
    supported_entities: Set[str]
    is_io_schema: bool
