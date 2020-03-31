from typing import Callable

from .conll_reader import get_conll_eval_data
from .wnut_reader import get_wnut_eval_data


class DataReaderRegistry:
    def __init__(self):
        self.registry = {}
        self.add_predefined_readers()

    def add_predefined_readers(self):
        self.add_reader(get_conll_eval_data, "conll")
        self.add_reader(get_wnut_eval_data, "wnut")

    def add_reader(self, reader: Callable, reader_tag: str):
        self.registry[reader_tag] = reader

    def get_reader(self, name: str) -> Callable:
        if name not in self.registry:
            raise ValueError(
                f"Found no reader of name {name}, available evaluation readers"
                f"are {self.registry.keys()}"
            )
        return self.registry[name]
