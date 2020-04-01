from typing import Callable, List

from nltk.tokenize import TreebankWordTokenizer

from .token_schema import Token

treebank_tokenizer = TreebankWordTokenizer()


def nltk_word_tokenizer(text: str) -> List[Token]:
    spans = list(treebank_tokenizer.span_tokenize(text))
    return [
        Token(text=text[span[0] : span[1]], start=span[0], end=span[1])
        for span in spans
    ]


class TokeniserRegistry:
    def __init__(self):
        self.registry = {}
        self.add_predefined_tokenisers()

    def add_predefined_tokenisers(self):
        self.add_tokeniser(nltk_word_tokenizer)

    def add_tokeniser(self, tokeniser: Callable):
        self.registry[tokeniser.__name__] = tokeniser
