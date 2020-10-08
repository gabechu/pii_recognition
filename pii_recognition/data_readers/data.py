from dataclasses import dataclass
from typing import List, TypeVar, Set, Generic, Optional
from pii_recognition.labels.schema import Entity

# Two kinds of entity labels
# 1. List[str] indicates every token has a label of entity type.
# 2. List[Entity] indicates every text-span has a label of entity type.
TEXT_LABELS = TypeVar("TEXT_LABELS", List[str], List[Entity])


@dataclass
class DataItem(Generic[TEXT_LABELS]):
    text: str
    true_labels: TEXT_LABELS
    pred_labels: Optional[TEXT_LABELS] = None


@dataclass
class Data:
    items: List[DataItem]
    supported_entities: Set[str]
    is_io_schema: bool
