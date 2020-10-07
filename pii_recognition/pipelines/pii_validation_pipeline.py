import os
from typing import Dict, List, Tuple

from pakkr import Pipeline, returns
from pii_recognition.constants import PROJECT_DIR
from pii_recognition.data_readers.data import Data
from pii_recognition.data_readers.presidio_fake_pii_reader import PresidioFakePiiReader
from pii_recognition.evaluation.character_level_evaluation import (
    compute_entity_precisions_for_prediction,
    compute_entity_recalls_for_ground_truth,
)
from pii_recognition.labels.schema import Entity
from pii_recognition.recognisers.comprehend_recogniser import ComprehendRecogniser


@returns(data=Data)
def read_benchmark_data(benchmark_data_file: str) -> Data:
    reader = PresidioFakePiiReader()
    return {"data": reader.build_data(benchmark_data_file)}


@returns(List)
def identify_pii_entities(data: Data, benchmark_data_file) -> Data:
    supported_entities = [
        "COMMERCIAL_ITEM",
        "DATE",
        "EVENT",
        "LOCATION",
        "ORGANIZATION",
        "OTHER",
        "PERSON",
        "QUANTITY",
        "TITLE",
    ]
    supported_languages = ["en"]
    recogniser = ComprehendRecogniser(supported_entities, supported_languages)

    for item in data.items[:5]:
        item.pred_label = recogniser.analyse(item.text, supported_entities)
    return data


@returns(Tuple)
def calculate_precisions_and_recalls(
    data: Data, label_mapping: Dict[str, int]
) -> Tuple[List, List]:
    data_precisions = []
    data_recalls = []
    for item in data.items[:5]:
        ent_precisions = compute_entity_precisions_for_prediction(
            len(item.text), item.true_label, item.pred_label, label_mapping
        )
        ent_recalls = compute_entity_recalls_for_ground_truth(
            len(item.text), item.true_label, item.pred_label, label_mapping
        )
        data_precisions.append(ent_precisions)
        data_recalls.append(ent_recalls)

    return data_precisions, data_recalls


@returns
def rollup_scores():
    ...


@returns
def log_scores_to_file():
    ...


def execute_evaluation_pipeline(config: Dict):
    eval_pipeline = Pipeline(
        read_benchmark_data, identify_pii_entities, calculate_precisions_and_recalls
    )

    return eval_pipeline(**config)


if __name__ == "__main__":
    config = {
        "benchmark_data_file": os.path.join(
            PROJECT_DIR,
            "datasets/predisio_fake_pii/generated_size_500_date_August_25_2020.json",
        ),
        "label_mapping": {
            # comprehend entity types
            "OTHER": 1,
            "DATE": 2,
            # benchmark entity types
            "US_SSN": 1,
            "IP_ADDRESS": 1,
            "PHONE_NUMBER": 1,
            "CREDIT_CARD": 1,
            "URL": 1,
            "EMAIL": 1,
            "BIRTHDAY": 2,
            # shared entity types
            "LOCATION": 3,
            "PERSON": 4,
            # ignored entity types
            "TITLE": 0,
            "ORGANIZATION": 0,
            "IBAN": 0,
            "NATIONALITY": 0,
        },
    }
    execute_evaluation_pipeline(config)
