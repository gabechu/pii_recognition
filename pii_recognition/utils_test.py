from unittest.mock import Mock, patch

from pii_recognition.utils import cached_property, is_ascending


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


def test_is_ascending():
    actual = is_ascending([1, 2, 3])
    assert actual is True

    actual = is_ascending([3, 2, 1])
    assert actual is False

    actual = is_ascending((1, 2, 3))
    assert actual is True

    actual = is_ascending(range(5))
    assert actual is True
