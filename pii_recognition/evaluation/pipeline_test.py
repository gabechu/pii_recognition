from unittest.mock import patch

from .pipeline import get_recogniser


@patch("pii_recognition.evaluation.pipeline.registry")
def test_get_recogniser(mock_regsitry):
    class Simple:
        pass

    class Complex:
        def __init__(self, a):
            self.a = a

    mock_regsitry.recogniser = {"Simple": Simple, "Complex": Complex}

    actual = get_recogniser("Simple")["recogniser"]
    assert isinstance(actual, Simple)

    actual = get_recogniser("Complex", {"a": "attr_a"})["recogniser"]
    assert isinstance(actual, Complex)
    assert actual.a == "attr_a"
