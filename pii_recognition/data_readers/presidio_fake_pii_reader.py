from typing import Dict, List, Set, Union

from pii_recognition.data_readers.data import Data, DataItem, Entity
from pii_recognition.utils import load_json_file


class PresidioFakePiiReader:
    def _get_supported_entities(self, data: List[Dict]) -> Set[str]:
        unique_labels = set()
        for item in data:
            spans = item["spans"]
            for span in spans:
                unique_labels.add(span["entity_type"])
        return unique_labels

    # TODO: test this function
    def _span_to_entity(self, span: Dict[str, Union[str, int]]) -> Entity:
        return Entity(span["entity_type"], span["start_position"], span["end_position"])

    def build_data(self, file_path: str) -> Data:
        data = load_json_file(file_path)
        supported_entities = self._get_supported_entities(data)

        items = []
        for item in data:
            entities = [self._span_to_entity(span) for span in item["spans"]]
            items.append(DataItem(text=item["full_text"], true_label=entities))

        return Data(
            items=items, supported_entities=supported_entities, is_io_schema=False,
        )
