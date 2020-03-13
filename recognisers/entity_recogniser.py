from abc import ABCMeta, abstractmethod
from typing import Any, List, TypeVar

from .recogniser_result import RecogniserResult


class EntityRecogniser(metaclass=ABCMeta):
    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        name: str = None,
        version: str = "0.0.1",
        **kwargs,
    ):
        if not name:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.version = version
        self.supported_entities = supported_entities
        self.supported_languages = supported_languages

        self._model = self.load_model()

    def validate_entities(self, asked_entities: List[str]):
        """Check whether asked entities are supported by the model."""
        assert all(
            [entity in self.supported_entities for entity in asked_entities]
        ), f"Only support {self.supported_entities}, but got {asked_entities}"

    def validate_languages(self, asked_languages: List[str]):
        """Check whether asked languages are supported by the model."""
        assert all(
            [language in self.supported_languages for language in asked_languages]
        ), f"Only support {self.supported_languages}, but got {asked_languages}"

    @abstractmethod
    def load_model(self) -> Any:
        ...

    @abstractmethod
    def analyse(self, text: str, entities: List[str]) -> List[RecogniserResult]:
        """Anotate asked entities in the text."""
        ...


Rec_co = TypeVar("Rec_co", bound=EntityRecogniser)
