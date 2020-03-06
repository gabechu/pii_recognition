from pathlib import PurePath
from typing import Callable, List, Tuple

from nltk.corpus.reader import ConllCorpusReader


def get_conll_eval_data(file_path: str, detokenizer: Callable) -> Tuple:
    path = PurePath(file_path)

    data = ConllCorpusReader(
        str(path.parents[0]), str(path.name), ["words", "pos", "ignore", "chunk"]
    )
    sent_features = list(data.iob_sents())

    sents = [detokenizer(sent2tokens(sent)) for sent in sent_features]
    labels = [sent2labels(sent) for sent in sent_features]

    return sents, labels


def sent2tokens(sent: List) -> List:
    return [token for token, postag, label in sent]


def sent2labels(sent: List) -> List:
    return [label for token, postag, label in sent]


def spacy_join_detokenzier(tokens: List) -> str:
    return " ".join(tokens)
