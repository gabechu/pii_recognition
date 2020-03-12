from .detokeniser import space_join_detokenzier


def test_space_join_detokenizer():
    tokens = ["Here", "is", "a", "test", "."]
    actual = space_join_detokenzier(tokens)
    assert actual == "Here is a test ."
