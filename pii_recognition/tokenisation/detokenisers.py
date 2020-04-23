from abc import ABCMeta, abstractmethod
from typing import Callable, List

from nltk.tokenize.treebank import TreebankWordDetokenizer


class Detokeniser(metaclass=ABCMeta):
    @abstractmethod
    def detokenise(self, tokens: List[str]) -> str:
        ...


def space_join_detokensier(tokens: List[str]) -> str:
    return " ".join(tokens)


def treebank_detokeniser(tokens: List[str]) -> str:
    return TreebankWordDetokenizer().detokenize(tokens)


class DetokeniserRegistry:
    def __init__(self):
        self.registry = {}
        self.add_predefined_detokenisers()

    def add_predefined_detokenisers(self):
        self.add_detokeniser(space_join_detokensier)
        self.add_detokeniser(treebank_detokeniser)

    def add_detokeniser(self, detokeniser: Callable):
        self.registry[detokeniser.__name__] = detokeniser
