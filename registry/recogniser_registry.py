from .registry import Registry

from recognisers.crf_recogniser import CrfRecogniser
from recognisers.first_letter_uppercase_recogniser import FirstLetterUppercaseRecogniser
from recognisers.flair_recogniser import FlairRecogniser
from recognisers.spacy_recogniser import SpacyRecogniser
from recognisers.stanza_recogniser import StanzaRecogniser


class RecogniserRegistry(Registry):
    def add_predefines(self):
        self.add_item(CrfRecogniser)
        self.add_item(FirstLetterUppercaseRecogniser)
        self.add_item(FlairRecogniser)
        self.add_item(SpacyRecogniser)
        self.add_item(StanzaRecogniser)
