import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

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


def load_json_file(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


# Any is not a precise signature but it's ergonomic in practice
def dump_to_json_file(obj: Any, path: str):
    with open(path, "w") as f:
        json.dump(obj, f)


def stringify_keys(data: Dict) -> Dict[str, Any]:
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = stringify_keys(value)

        if not isinstance(key, str):
            data[str(key)] = value
            del data[key]
    return data
