from typing import Callable, List, Tuple
from label.mapping import map_bio_to_io_labels


def get_wnut_eval_data(
    file_path: str, detokenizer: Callable[[List[str]], str]
) -> Tuple[List[str], List[List[str]]]:
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
            else:  # hit empty line
                sents.append(detokenizer(sentence_tokens))
                labels.append(map_bio_to_io_labels(sentence_entities))
                sentence_tokens = []
                sentence_entities = []

    # clear up
    if sentence_tokens and sentence_entities:
        sents.append(detokenizer(sentence_tokens))
        labels.append(map_bio_to_io_labels(sentence_entities))
    return sents, labels
