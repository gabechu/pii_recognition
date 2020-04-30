from typing import List, Tuple, Dict

from pii_recognition.labels.mapping import map_bio_to_io_labels
from .reader import Reader
from pii_recognition.tokenisation import detokeniser_registry


class WnutReader(Reader):
    def __init__(self, detokeniser_setup: Dict):
        self._detokeniser = detokeniser_registry.create_instance(
            name=detokeniser_setup["name"], config=detokeniser_setup.get("config")
        )

    def get_test_data(self, file_path: str) -> Tuple[List[str], List[List[str]]]:
        """
        Label types of WNUT 2017 evaluation data:
            I-person
            I-location
            I-corporation
            I-product
            I-creative-work
            I-group
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
                    # hit empty line and next line is start of a new sentence
                    sents.append(self._detokeniser.detokenise(sentence_tokens))
                    labels.append(map_bio_to_io_labels(sentence_entities))
                    # refresh containers
                    sentence_tokens = []
                    sentence_entities = []

        # process the last one
        if sentence_tokens and sentence_entities:
            sents.append(self._detokeniser.detokenise(sentence_tokens))
            labels.append(map_bio_to_io_labels(sentence_entities))
        return sents, labels
