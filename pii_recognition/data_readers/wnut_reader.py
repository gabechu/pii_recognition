from typing import List

from pii_recognition.labels.mapping import map_bio_to_io_labels
from pii_recognition.tokenisation.detokenisers import Detokeniser

from .reader import Data, Reader


class WnutReader(Reader):
    def __init__(self, detokeniser: Detokeniser):
        self._detokeniser = detokeniser

    def get_test_data(
        self, file_path: str, supported_entities: List[str], is_io_schema: bool = True
    ) -> Data:
        """
        Read WNUT type of data.
        """
        sents = []
        labels = []
        sentence_tokens = []
        sentence_entities = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.split()
                if data:
                    token, entity_tag = data
                    sentence_tokens.append(token)
                    sentence_entities.append(entity_tag)
                else:
                    # hit empty line and the next line is the start of a new sentence
                    # flush the collected sentence and labels
                    if is_io_schema:
                        processed_labels = map_bio_to_io_labels(sentence_entities)
                    else:
                        processed_labels = sentence_entities
                    self._validate_entity(
                        set(processed_labels), set(supported_entities)
                    )

                    labels.append(processed_labels)
                    sents.append(self._detokeniser.detokenise(sentence_tokens))

                    # refresh containers
                    sentence_tokens = []
                    sentence_entities = []

        # process the last one
        if sentence_tokens and sentence_entities:
            sents.append(self._detokeniser.detokenise(sentence_tokens))
            labels.append(map_bio_to_io_labels(sentence_entities))
        return Data(sents, labels, supported_entities, is_io_schema,)
