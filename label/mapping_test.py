from .mapping import map_labels, mask_labels, map_bio_to_io_labels


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


def test_map_bio_to_io_labels():
    bio_labels = [
        "O",
        "B-LOC",
        "I-LOC",
        "O",
        "B-PER",
        "O",
        "B-ORG",
        "B-ORG",
        "I-ORG",
        "O",
    ]
    actual = map_bio_to_io_labels(bio_labels)
    assert actual == [
        "O",
        "I-LOC",
        "I-LOC",
        "O",
        "I-PER",
        "O",
        "I-ORG",
        "I-ORG",
        "I-ORG",
        "O",
    ]
