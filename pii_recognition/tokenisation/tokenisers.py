from abc import ABCMeta, abstractmethod
from typing import List

from nltk.tokenize import TreebankWordTokenizer as TreebankWordTokenizer_

from pii_recognition.utils import cached_property

from .token_schema import Token


class Tokeniser(metaclass=ABCMeta):
    @abstractmethod
    def tokenise(self, text: str) -> List[Token]:
        ...


class TreebankWordTokeniser(Tokeniser):
    @cached_property
    def _engine(self) -> TreebankWordTokenizer_:
        return TreebankWordTokenizer_()

    def tokenise(self, text: str) -> List[Token]:
        # spans is a list of tuples e.g. [(0:5), (6:10)]
        # using (start_index: end_index) to specify a word
        spans = list(self._engine.span_tokenize(text))
        return [
            Token(text=text[span[0] : span[1]], start=span[0], end=span[1])
            for span in spans
        ]
