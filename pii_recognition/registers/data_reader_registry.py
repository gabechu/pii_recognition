from constants import CONLL, WNUT
from data_reader.conll_reader import get_conll_eval_data
from data_reader.wnut_reader import get_wnut_eval_data

from .registry import Registry


class DataReaderRegistry(Registry):
    def add_predefines(self):
        self[CONLL] = get_conll_eval_data
        self[WNUT] = get_wnut_eval_data
