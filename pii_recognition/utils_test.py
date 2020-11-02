import json
import os
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, mock_open, patch

from pii_recognition.utils import (
    cached_property,
    dump_to_json_file,
    dump_yaml_file,
    is_ascending,
    load_json_file,
    load_yaml_file,
    stringify_keys,
    write_iterable_to_file,
)


def json_dumps():
    return json.dumps(
        [{"name": "John", "location": "AU"}, {"name": "Mia", "location": "AU"}]
    )


@patch("builtins.open", new_callable=mock_open)
def test_write_iterable_to_file(mock_open_file):
    write_iterable_to_file([1, 2], "fake_path")
    handle = mock_open_file()
    handle.write.assert_has_calls([call("1\n"), call("2\n")])


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


def test_load_yaml_file():
    with patch("builtins.open", mock_open(read_data="TEST-KEY: TEST-VALUE\n")):
        data = load_yaml_file("fake_path")
    assert data == {"TEST-KEY": "TEST-VALUE"}


def test_dump_yaml_file():
    with patch("builtins.open", mock_open()) as m:
        dump_yaml_file("fake_path", {"key": "value"})
        handle = m()
        handle.write.assert_has_calls(
            [call("key"), call(":"), call(" "), call("value"), call("\n")]
        )


@patch("builtins.open", new_callable=mock_open, read_data=json_dumps())
def test_load_json_file(mock_file):
    data = load_json_file("any_path/file.json")
    assert data == [
        {"name": "John", "location": "AU"},
        {"name": "Mia", "location": "AU"},
    ]


@patch("builtins.open", new_callable=mock_open)
def test_dump_to_json_file(mock_open_file):
    dump_to_json_file("test_dump", "fake_path")

    handle = mock_open_file()
    handle.write.assert_called_once_with('"test_dump"')


def test_dump_and_read_json_file():
    obj = [
        {"test_key_1": "test_value_1"},
        {"test_key_2": {"test_sub_key_1": "test_sub_value_1"}},
    ]

    with TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, "test.json")
        dump_to_json_file(obj, file_path)
        actual = load_json_file(file_path)
    assert actual == obj


def test_stringify_keys_for_int():
    actual = stringify_keys({1: 2})
    assert actual == {"1": 2}


def test_stringify_keys_for_float():
    actual = stringify_keys({1.0: 2})
    assert actual == {"1.0": 2}


def test_stringify_keys_for_bool():
    actual = stringify_keys({True: 1})
    assert actual == {"True": 1}


def test_stringify_keys_for_tuple():
    actual = stringify_keys({(1, 2): 3})
    assert actual == {"(1, 2)": 3}


def test_stringify_keys_for_range():
    actual = stringify_keys({range(3): 3})
    assert actual == {"range(0, 3)": 3}


def test_stringify_keys_for_frozenset():
    actual = stringify_keys({frozenset([1, 2]): 3})
    assert actual == {"frozenset({1, 2})": 3}


def test_stringify_keys_for_bytes():
    actual = stringify_keys({b"\x00\x10": 1})
    assert actual == {"b'\\x00\\x10'": 1}


def test_stringify_keys_for_instance():
    class MyClass:
        ...

    instance = MyClass()
    actual = stringify_keys({instance: 1})
    assert isinstance(list(actual.keys())[0], str)
    assert list(actual.values()) == [1]


def test_stringify_keys_for_nested():
    actual = stringify_keys({(1, 2): {1: 1, 2: 2}, 3: 3})
    assert actual == {"(1, 2)": {"1": 1, "2": 2}, "3": 3}
