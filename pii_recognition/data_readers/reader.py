from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Set


# TODO: replace Data with the one from data.py
@dataclass
class Data:
    # a recogniser takes a string as input so need a setence
    # instead of a list of tokens
    sentences: List[str]
    # labels are token based
    labels: List[List[str]]
    supported_entities: List[str]
    is_io_schema: bool


class Reader(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        ...

    def _validate_entity(self, loaded_entities: Set[str], supported_entities: Set[str]):
        if "O" not in supported_entities:
            loaded_entities = loaded_entities - {"O"}

        unsupported = loaded_entities - supported_entities
        if unsupported:
            raise ValueError(
                f"Found unsupported entity {unsupported} in data. "
                f"You may need to update your supported entity list."
            )

    @abstractmethod
    def get_test_data(
        self, file_path: str, supported_entities: List[str], is_io_schema: bool = True
    ) -> Data:
        """
        Read test data and split into features and labels. Features are inputs
        to a model and labels are the ground truths.
        """
        ...
