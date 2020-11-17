import json
from typing import Any, Dict, Iterable, Optional, Sequence, Type

import yaml


def write_iterable_to_file(iterable: Iterable, file_path: str):
    with open(file_path, "w") as f:
        for elem in iterable:
            f.write(str(elem) + "\n")


class cached_property(property):  # class name follows the convention of property
    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
        if self.fget is None:
            raise AttributeError("unreadable attribute")

        if obj is None:
            return self

        name = self.fget.__name__
        if name in obj.__dict__:
            return obj.__dict__[name]  # cached already
        else:
            value = self.fget(obj)  # type: ignore
            obj.__dict__[name] = value  # saving back to object
            return value


def is_ascending(sequence: Sequence) -> bool:
    return all(sequence[i] < sequence[i + 1] for i in range(len(sequence) - 1))


def load_yaml_file(path: str) -> Optional[Dict]:
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def dump_yaml_file(path: str, data: Any):
    with open(path, "w") as stream:
        yaml.dump(data, stream)


# Type hint for json.load has not been supported because of recursive types
# found details here https://github.com/python/typing/issues/182
def load_json_file(path: str):
    with open(path, "r") as f:
        return json.load(f)


# Any is not a precise signature but it's ergonomic in practice
def dump_to_json_file(obj: Any, path: str):
    # TODO: enable write to directories that do not exist
    with open(path, "w") as f:
        json.dump(obj, f)


def stringify_keys(data: Dict) -> Dict[str, Any]:
    stringify_dict = dict()
    for key, value in data.items():
        if not isinstance(key, str):
            new_key = str(key)
        else:
            new_key = key

        if isinstance(value, dict):
            stringify_dict[new_key] = stringify_keys(value)
        else:
            stringify_dict[new_key] = value

    return stringify_dict


class TextIndexer:
    """Convert index in one encoding to index in another encoding."""

    def __init__(self, text: str):
        self.text = text
        self._byte_to_utf8_mapping: Optional[Dict[int, int]] = None

    @property
    def byte_to_utf8_mapping(self) -> Dict[int, int]:
        if not self._byte_to_utf8_mapping:
            byte_index = utf8_index = 0
            mapping = {byte_index: utf8_index}
            for char in self.text:
                byte_index += len(char.encode())
                utf8_index += 1
                mapping[byte_index] = utf8_index

            self._byte_to_utf8_mapping = mapping

        return self._byte_to_utf8_mapping

    def byte_index_to_utf8_index(self, byte_index: int) -> int:
        try:
            return self.byte_to_utf8_mapping[byte_index]
        except KeyError:
            # The only usage now is Google NL models and so far it does
            # not cause any failures on index conversion. We may not consider
            # logics on handling failures until we encouter such cases.
            raise Exception(
                f"Index {byte_index} is an invalid boundary converting to UTF8."
            )
