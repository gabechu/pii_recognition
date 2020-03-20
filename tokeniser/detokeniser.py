from typing import List
from nltk.tokenize.treebank import TreebankWordDetokenizer


def space_join_detokensier(tokens: List[str]) -> str:
    return " ".join(tokens)


def treebank_detokeniser(tokens: List[str]) -> str:
    return TreebankWordDetokenizer().detokenize(tokens)
