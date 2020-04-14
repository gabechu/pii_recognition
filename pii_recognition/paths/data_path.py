from .path import Path


class DataPath(Path):
    pattern_str: str = r".*datasets/(?P<data_name>[a-zA-Z]+)(?P<version>\d+)/"
