from typing import List

import pytest

from .data import Data


def test_data_for_texts_labels_length_mismatch():
    texts = ["A tribute to Joshua Lewis", "It's like that since 12/17/1967"]
    labels: List = []

    with pytest.raises(AssertionError) as err:
        Data(
            texts=texts,
            labels=labels,
            supported_entities={"PERSON"},
            is_io_schema=False,
        )
    assert str(err.value) == (
        "Texts length does not match with labels length: "
        "texts length is 2 and labels length is 0"
    )
