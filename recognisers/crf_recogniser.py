from typing import Callable, List

from pycrfsuite import Tagger

from features.word_to_features import word2features

from .entity_recogniser import EntityRecogniser
from .recogniser_result import RecogniserResult


class CrfRecogniser(EntityRecogniser):
    def __init__(
        self,
        supported_entities: List,
        supported_languages: List,
        model_path: str,
        tokenizer: Callable,
    ):
        self._model_path = model_path
        self._tokenizer = tokenizer
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    def load_model(self) -> Tagger:
        tagger = Tagger()
        tagger.open(self._model_path)
        return tagger

    def preprocess_text(self, text: str) -> List:
        return self._tokenizer(text)

    def build_features(self, tokenised_sentence: List) -> List:
        return [
            word2features(tokenised_sentence, i) for i in range(len(tokenised_sentence))
        ]

    def analyze(self, text: str, entities: List) -> List:
        self.validate_entities(entities)
        # TODO: validate languages

        preprocessed_text = self.preprocess_text(text)
        tokens = [token.data for token in preprocessed_text]
        features = self.build_features(tokens)
        entity_tags = self._model.tag(features)

        assert len(entity_tags) == len(tokens) == len(preprocessed_text)

        results = []
        for i in range(len(entity_tags)):
            entity_type = entity_tags[i]
            if entity_type in entities:
                results.append(
                    RecogniserResult(
                        entity_type=entity_type,
                        start=preprocessed_text[i].start,
                        end=preprocessed_text[i].end,
                    )
                )
        return results
