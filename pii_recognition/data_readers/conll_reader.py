from pathlib import PurePath
from typing import List, Tuple

from nltk.corpus.reader import ConllCorpusReader

from pii_recognition.tokenisation.detokenisers import Detokeniser

from .reader import Reader


def _sent2tokens(sent: List[Tuple[str, str, str]]) -> List[str]:
    return [token for token, postag, label in sent]


def _sent2labels(sent: List[Tuple[str, str, str]]) -> List[str]:
    return [label for token, postag, label in sent]


class ConllReader(Reader):
    def __init__(self, detokeniser: Detokeniser):
        self._detokeniser = detokeniser

    def _get_corpus(self, file_path: str) -> ConllCorpusReader:
        path = PurePath(file_path)
        return ConllCorpusReader(
            root=str(path.parents[0]),
            fileids=str(path.name),
            columntypes=["words", "pos", "ignore", "chunk"],
        )

    def get_test_data(self, file_path: str) -> Tuple[List[str], List[List[str]]]:
        """
        Label types of CONLL 2003 evaluation data:
            I-PER
            I-LOC
            I_ORG
            I-MISC
        """
        data = self._get_corpus(file_path)

        sent_features = list(data.iob_sents())
        sent_features = [x for x in sent_features if x]  # remove empty features

        sents = [
            self._detokeniser.detokenise(_sent2tokens(sent)) for sent in sent_features
        ]
        labels = [_sent2labels(sent) for sent in sent_features]

        return sents, labels
