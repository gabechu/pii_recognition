from pii_recognition.utils import cached_property
from unittest.mock import patch, Mock


def test_cached_property():
    class Toy:
        @cached_property
        def a(self):
            return "value_a"

    # test 1: attribute is there
    actual = Toy()
    actual.a
    assert "a" in actual.__dict__
    assert actual.a == "value_a"
    del actual

    # test 2: called once
    mock_fget = Mock()
    mock_fget.__name__ = "mock_fget"
    with patch.object(cached_property, attribute="fget", new=mock_fget):
        actual = Toy()
        actual.a
        actual.a
        mock_fget.assert_called_once()
