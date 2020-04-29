from abc import ABCMeta, abstractmethod
from typing import List

from nltk.tokenize.treebank import TreebankWordDetokenizer as TreebankWordDetokenizer_

from pii_recognition.utils import cached_property


class Detokeniser(metaclass=ABCMeta):
    @abstractmethod
    def detokenise(self, tokens: List[str]) -> str:
        ...


class SpaceJoinDetokeniser(Detokeniser):
    def detokenise(self, tokens: List[str]) -> str:
        return " ".join(tokens)


class TreebankWordDetokeniser(Detokeniser):
    @cached_property
    def _engine(self) -> TreebankWordDetokenizer_:
        return TreebankWordDetokenizer_()

    def detokenise(self, tokens: List[str]) -> str:
        return self._engine.detokenize(tokens)
