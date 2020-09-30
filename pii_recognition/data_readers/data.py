from dataclasses import dataclass
from typing import List, TypeVar, Set, Generic
from pii_recognition.labels.schema import Entity

# Two kinds of entity labels
# 1. List[str] indicates every token has a label.
# 2. List[Entity] indicates every text-span has a label.
TEXT_LABELS = TypeVar("TEXT_LABELS", List[str], List[Entity])


@dataclass
class Data(Generic[TEXT_LABELS]):
    texts: List[str]
    labels: List[TEXT_LABELS]
    supported_entities: Set[str]
    is_io_schema: bool

    def __post_init__(self):
        assert len(self.texts) == len(self.labels), (
            f"Texts length does not match with labels length: texts length is "
            f"{len(self.texts)} and labels length is {len(self.labels)}"
        )
