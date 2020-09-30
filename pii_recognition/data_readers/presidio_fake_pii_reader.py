from typing import Dict, List, Set

from pii_recognition.data_readers.data import Data
from pii_recognition.utils import load_json_file


class PresidioFakePiiReader:
    def _get_supported_entities(self, data: List[Dict]) -> Set[str]:
        unique_labels = set()
        for item in data:
            spans = item["spans"]
            for span in spans:
                unique_labels.add(span["entity_type"])
        return unique_labels

    def build_data(self, file_path: str) -> Data:
        data = load_json_file(file_path)
        supported_entities = self._get_supported_entities(data)

        return Data(
            texts=[item["full_text"] for item in data],
            labels=[item["spans"] for item in data],
            supported_entities=supported_entities,
            is_io_schema=False,
        )
