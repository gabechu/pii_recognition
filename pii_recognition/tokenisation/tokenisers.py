from abc import ABCMeta, abstractmethod
from typing import List

from nltk.tokenize import TreebankWordTokenizer

from .token_schema import Token


class Tokeniser(metaclass=ABCMeta):
    @abstractmethod
    def tokenise(self, text: str) -> List[Token]:
        ...


treebank_tokenizer = TreebankWordTokenizer()


def nltk_word_tokenizer(text: str) -> List[Token]:
    spans = list(treebank_tokenizer.span_tokenize(text))
    return [
        Token(text=text[span[0] : span[1]], start=span[0], end=span[1])
        for span in spans
    ]
