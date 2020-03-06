from typing import List
from .token import Token

from nltk.tokenize import TreebankWordTokenizer

treebank_tokenizer = TreebankWordTokenizer()


def word_tokenizer(text: str) -> List:
    spans = list(treebank_tokenizer.span_tokenize(text))
    return [
        Token(data=text[span[0] : span[1]], start=span[0], end=span[1])
        for span in spans
    ]
