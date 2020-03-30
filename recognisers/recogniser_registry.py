from typing import Type

from .entity_recogniser import EntityRecogniser


class RecogniserRegistry:
    def __init__(self):
        self.registry = {}

    def add_recogniser(self, recogniser: Type[EntityRecogniser]):
        # TODO: myerror
        # https://github.com/python/mypy/issues/3728
        self.registry[recogniser.__name__] = recogniser  # type: ignore

    def get_recogniser(self, name: str) -> EntityRecogniser:
        if name not in self.registry:
            raise ValueError(
                f"Recogniser not found, available recognisers are"
                f"{self.registry.keys()}"
            )

        return self.registry[name]
