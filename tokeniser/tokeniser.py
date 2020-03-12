from typing import List

from nltk.tokenize import TreebankWordTokenizer

from .token import Token

treebank_tokenizer = TreebankWordTokenizer()


def nltk_word_tokenizer(text: str) -> List[Token]:
    spans = list(treebank_tokenizer.span_tokenize(text))
    return [
        Token(text=text[span[0] : span[1]], start=span[0], end=span[1])
        for span in spans
    ]
