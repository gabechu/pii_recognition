from .mapping import map_labels


def test_map_labels():
    actual = map_labels(
        type_A_labels=["O", "O", "PER", "LOC"],
        A2B_mapping={"PER": "PERSON", "LOC": "LOCATION"},
    )

    assert actual == ["O", "O", "PERSON", "LOCATION"]
