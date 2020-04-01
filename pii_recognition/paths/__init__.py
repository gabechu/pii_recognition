from .path import create_path_subclass


DataPath = create_path_subclass(
    "DataPath", "datasets/(?P<data_name>[a-zA-Z]+)(?P<version>\d+)/"
)
