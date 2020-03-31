from typing import Callable

from .conll_reader import get_conll_eval_data
from .wnut_reader import get_wnut_eval_data


class DataReaderRegistry:
    def __init__(self):
        self.registry = {}
        self.add_predefined_readers()

    def add_predefined_readers(self):
        self.add_reader(get_conll_eval_data)
        self.add_reader(get_wnut_eval_data)

    def add_reader(self, reader: Callable):
        self.registry[reader.__name__] = reader

    def get_reader(self, reader_name: str) -> Callable:
        if reader_name not in self.registry:
            raise ValueError(
                f"Found no reader of name {reader_name}, available evaluation readers"
                f"are {self.registry.keys()}"
            )
        return self.registry[reader_name]
