from .detokenisers import SpaceJoinDetokeniser, TreebankWordDetokeniser


def test_SpaceJoinDetokeniser():
    tokens = ["Here", "is", "a", "test", "."]
    detokeniser = SpaceJoinDetokeniser()
    actual = detokeniser.detokenise(tokens)
    assert actual == "Here is a test ."


def test_treebank_detokeniser():
    tokens = ["Here", "is", "a", "test", "."]
    detokeniser = TreebankWordDetokeniser()
    actual = detokeniser.detokenise(tokens)
    assert actual == "Here is a test."
