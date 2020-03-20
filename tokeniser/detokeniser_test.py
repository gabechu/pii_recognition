from .detokeniser import space_join_detokensier, treebank_detokeniser


def test_space_join_detokeniser():
    tokens = ["Here", "is", "a", "test", "."]
    actual = space_join_detokensier(tokens)
    assert actual == "Here is a test ."


def test_treebank_detokeniser():
    tokens = ["Here", "is", "a", "test", "."]
    actual = treebank_detokeniser(tokens)
    assert actual == "Here is a test."
