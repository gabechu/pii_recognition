from abc import ABCMeta, abstractmethod
from typing import List, Tuple


class Reader(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def get_evaluation_data(self, file_path: str) -> Tuple[List[str], List[List[str]]]:
        """
        Read evaluation data and split into features and labels.
        """
        ...
