from typing import Callable, List

from pycrfsuite import Tagger

from features.word_to_features import word2features
from tokeniser.token import Token

from .entity_recogniser import EntityRecogniser
from label.label_schema import SpanLabel


class CrfRecogniser(EntityRecogniser):
    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        model_path: str,
        tokenizer: Callable[[str], List[Token]],
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

    def preprocess_text(self, text: str) -> List[Token]:
        return self._tokenizer(text)

    def build_features(self, tokenised_sentence: List[str]) -> List[List[str]]:
        return [
            word2features(tokenised_sentence, i) for i in range(len(tokenised_sentence))
        ]

    @staticmethod
    def _get_span_labels(tokens: List[Token], tags: List[str]) -> List[SpanLabel]:
        assert len(tokens) == len(tags), (
            f"Length mismatch, where len(tokens)={len(tokens)} and "
            f"len(tags)={len(tags)}"
        )

        span_labels = []
        segment_start = tokens[0].start
        segment_end = tokens[0].end

        if len(tags) == 1:
            return [SpanLabel(tags[0], segment_start, segment_end)]

        for i in range(1, len(tags)):
            if tags[i] == tags[i - 1]:
                segment_end = tokens[i].end
            else:
                span_labels.append(SpanLabel(tags[i - 1], segment_start, segment_end))
                segment_start = tokens[i].start
                segment_end = tokens[i].end

            if i == len(tags) - 1:
                segment_end = tokens[i].end
                span_labels.append(SpanLabel(tags[i], segment_start, segment_end))
        return span_labels

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        self.validate_entities(entities)
        # TODO: validate languages

        preprocessed_text = self.preprocess_text(text)
        tokens = [token.text for token in preprocessed_text]
        features = self.build_features(tokens)
        entity_tags = self._model.tag(features)

        assert len(entity_tags) == len(tokens) == len(preprocessed_text)

        spans = CrfRecogniser._get_span_labels(preprocessed_text, entity_tags)
        return [span for span in spans if span.entity_type in entities]
