from typing import Dict, List, Set
from typing_extensions import TypedDict

from pii_recognition.data_readers.data import Data, DataItem, Entity
from pii_recognition.utils import load_json_file

# mypy assumes that Dict is homogeneous, however, we
# need it to be heterogenous thus use TypedDict
PresidioSpan = TypedDict(
    "PresidioSpan",
    {
        "entity_type": str,
        "entity_value": str,
        "start_position": int,
        "end_position": int,
    },
)


class PresidioFakePiiReader:
    def _get_supported_entities(self, data: List[Dict]) -> Set[str]:
        unique_labels = set()
        for item in data:
            spans = item["spans"]
            for span in spans:
                unique_labels.add(span["entity_type"])
        return unique_labels

    def _span_to_entity(self, span: PresidioSpan) -> Entity:
        return Entity(span["entity_type"], span["start_position"], span["end_position"])

    def build_data(self, file_path: str) -> Data:
        data = load_json_file(file_path)
        supported_entities = self._get_supported_entities(data)

        items = []
        for item in data:
            entities = [self._span_to_entity(span) for span in item["spans"]]
            items.append(DataItem(text=item["full_text"], true_labels=entities))

        return Data(
            items=items, supported_entities=supported_entities, is_io_schema=False,
        )
