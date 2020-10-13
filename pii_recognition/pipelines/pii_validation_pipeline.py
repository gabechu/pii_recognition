from typing import Dict, List, Set, Optional

from pakkr import returns
from pii_recognition.data_readers.data import Data
from pii_recognition.data_readers.presidio_fake_pii_reader import PresidioFakePiiReader
from pii_recognition.evaluation.character_level_evaluation import (
    TextScore,
    build_label_mapping,
    compute_entity_precisions_for_prediction,
    compute_entity_recalls_for_ground_truth,
)
from pii_recognition.recognisers import registry as recogniser_registry
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser


@returns(Data)
def read_benchmark_data(benchmark_data_file: str) -> Data:
    reader = PresidioFakePiiReader()
    return reader.build_data(benchmark_data_file)


@returns(Data)
def identify_pii_entities(
    data: Data, recogniser_name: str, recogniser_params: Dict
) -> Data:
    recogniser: EntityRecogniser = recogniser_registry.create_instance(
        recogniser_name, recogniser_params
    )

    for item in data.items:
        item.pred_labels = recogniser.analyse(item.text, recogniser.supported_entities)
    return data


@returns(List)
def calculate_precisions_and_recalls(
    data: Data,
    grouped_targeted_labels: List[Set[str]],
    nontargeted_labels: Optional[Set[str]] = None,
) -> List[TextScore]:
    label_mapping = build_label_mapping(grouped_targeted_labels, nontargeted_labels)

    scores = []
    for item in data.items:
        if item.pred_labels:
            pred_labels = item.pred_labels
        else:  # pred_labels could be None
            pred_labels = []

        ent_precisions = compute_entity_precisions_for_prediction(
            len(item.text), item.true_labels, pred_labels, label_mapping
        )
        ent_recalls = compute_entity_recalls_for_ground_truth(
            len(item.text), item.true_labels, pred_labels, label_mapping
        )
        ticket_score = TextScore(precisions=ent_precisions, recalls=ent_recalls)
        scores.append(ticket_score)

    return scores
