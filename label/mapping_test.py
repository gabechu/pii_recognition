from .mapping import map_labels, mask_labels


def test_map_labels():
    actual = map_labels(
        type_A_labels=["O", "O", "PER", "LOC"],
        A2B_mapping={"PER": "PERSON", "LOC": "LOCATION"},
    )

    assert actual == ["O", "O", "PERSON", "LOCATION"]


def test_mask_labels():
    actual = mask_labels(
        input_labels=["O", "DATE", "PER", "LOC"], non_mask_labels=["PER", "LOC"]
    )
    assert actual == ["O", "O", "PER", "LOC"]

    actual = mask_labels(
        input_labels=["O", "DATE", "PER", "LOC"], non_mask_labels=["PERSON", "LOCATION"]
    )
    assert actual == ["O", "O", "O", "O"]
