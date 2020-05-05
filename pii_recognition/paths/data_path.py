from .path import Path


class DataPath(Path):
    pattern_str: str = r".*datasets/(?P<data_name>[a-zA-Z]+)(?P<version>\d+)/"

    @property
    def reader_name(self):
        # available readers are defined at `data_readers` folder
        mapping = {"conll": "ConllReader", "wnut": "WnutReader"}

        if self.data_name not in mapping:
            raise NameError(f"No reader found to process {self.data_name} dataset")
        return mapping[self.data_name]
