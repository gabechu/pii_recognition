from .path import Path


class DataPath(Path):
    pattern_str: str = r".*datasets/(?P<data_name>[a-zA-Z]+)(?P<version>\d+)/"

    @property
    def _lookup(self):
        """A lookup table helps find a reader of a given data path."""
        # Find all available readers in the `data_reader` folder
        return {"conll": "ConllReader", "wnut": "WnutReader"}

    @property
    def reader_name(self):
        if self.data_name not in self._lookup:
            raise NameError(
                f"No reader found to process {self.data_name} dataset. "
                f"Update `_lookup` property for adding additional readers."
            )
        return self._lookup[self.data_name]
