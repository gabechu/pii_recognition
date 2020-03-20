from .detokeniser import space_join_detokensier


def test_space_join_detokeniser():
    tokens = ["Here", "is", "a", "test", "."]
    actual = space_join_detokensier(tokens)
    assert actual == "Here is a test ."
