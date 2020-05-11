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
