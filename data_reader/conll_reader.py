from pathlib import PurePath
from typing import Callable, List, Tuple

from nltk.corpus.reader import ConllCorpusReader


def get_conll_eval_data(
    file_path: str, detokenizer: Callable[[List[str]], str]
) -> Tuple[List[str], List[List[str]]]:
    """
    Label types of CONLL 2003 evaluation data:
        I-PER
        I-LOC
        I_ORG
        I-MISC
    """
    path = PurePath(file_path)

    data = ConllCorpusReader(
        str(path.parents[0]), str(path.name), ["words", "pos", "ignore", "chunk"]
    )
    sent_features = list(data.iob_sents())
    sent_features = [x for x in sent_features if x]  # remove empty

    sents = [detokenizer(sent2tokens(sent)) for sent in sent_features]
    labels = [sent2labels(sent) for sent in sent_features]

    return sents, labels


def sent2tokens(sent: List[Tuple[str, str, str]]) -> List[str]:
    return [token for token, postag, label in sent]


def sent2labels(sent: List[Tuple[str, str, str]]) -> List[str]:
    return [label for token, postag, label in sent]
