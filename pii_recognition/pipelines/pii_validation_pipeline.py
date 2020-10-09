from typing import Dict

from pakkr import returns
from pii_recognition.data_readers.data import Data
from pii_recognition.data_readers.presidio_fake_pii_reader import PresidioFakePiiReader
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
