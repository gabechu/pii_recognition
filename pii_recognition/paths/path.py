import re
from typing import Match, Type


class Path:
    _pattern_str = None

    def __init__(self, path: str):
        self.path = path
        self._pattern_to_attrs(self.get_pattern())

    def get_pattern(self) -> Match:
        if not self._pattern_str:
            raise AttributeError("Pattern has not been defined.")

        return re.match(self._pattern_str, self.path)

    def _pattern_to_attrs(self, pattern: Match):
        for key, value in pattern.groupdict().items():
            setattr(self, key, value)


def create_path_subclass(cls_name: str, pattern: str) -> Type[Path]:
    return type(cls_name, (Path,), {"_pattern_str": pattern})
