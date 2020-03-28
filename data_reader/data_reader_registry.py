from typing import Callable

from .conll_reader import get_conll_eval_data
from .wnut_reader import get_wnut_eval_data


def get_eval_reader(name: str) -> Callable:
    available_eval_reader = {
        "conll2003": get_conll_eval_data,
        "wnut2017": get_wnut_eval_data,
    }

    assert (
        name in available_eval_reader
    ), f"Found no reader of name {name}, available evaluation readers are {available_eval_reader.keys()}"

    return available_eval_reader[name]
