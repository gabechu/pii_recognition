from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from recognisers.entity_recogniser import Rec_co
from tokenizers.token import Token

from .evaluation_result import EvaluationResult
from .label import Label, map_labels
from .metrics import compute_f_beta
from .prediction_error import SampleError, TokenError
from .span_to_token import span_labels_to_token_labels


class ModelEvaluator:
    def __init__(
        self,
        recogniser: Rec_co,
        target_entities: List[str],
        tokeniser: Callable[[str], List[Token]],
        entity_mapping: Optional[Dict[str, str]] = None,
    ):
        self.recogniser = recogniser
        self.target_entities = target_entities
        self.tokeniser = tokeniser
        self.entity_mapping = entity_mapping

    def predict_token_based_entities(self, text: str) -> List[str]:
        recognised_entities = self.recogniser.analyze(text, self.target_entities)
        tokens = self.tokeniser(text)
        return span_labels_to_token_labels(tokens, recognised_entities)

    def _compare(
        self, text: str, annotations: List[str], predictions: List[str]
    ) -> Tuple[Counter, SampleError]:
        """
        Given a sample of text, compare the ground truth entity labels dennoted by
        annotation and predicted entity labels denoted by predictions. Count the
        occurrence and find mistakes.
        """
        # annotation may use a different label schema, if so
        # use mapping to convert predictions to that schema for comparison
        if self.entity_mapping:
            predictions = map_labels(predictions, self.entity_mapping)

        label_pair_counter = Counter()
        if len(annotations) != len(predictions):
            return (
                label_pair_counter,
                SampleError(token_errors=[], full_text=text, length_mismatch=True),
            )

        sample_error = SampleError(
            token_errors=[], full_text=text, length_mismatch=False
        )
        tokens = self.tokeniser(text)

        for i in range(len(annotations)):
            label_pair_counter[Label(annotations[i], predictions[i])] += 1
            # log mistakes
            if annotations[i] != predictions[i]:
                sample_error.token_errors.append(
                    TokenError(
                        annotation=annotations[i],
                        prediction=predictions[i],
                        token=tokens[i].text,
                    )
                )

        return label_pair_counter, sample_error

    def evaluate_sample(self, text: str, annotations: List[str]) -> EvaluationResult:
        predictions = self.predict_token_based_entities(text)
        label_pair_counter, sample_error = self._compare(text, annotations, predictions)
        return EvaluationResult(
            label_pair_counter=label_pair_counter, mistakes=sample_error
        )

    def evaulate_all(
        self, texts: List[str], annotations: List[List[str]]
    ) -> List[EvaluationResult]:
        assert len(texts) == len(annotations), (
            f"The number of texts: {len(texts)} mismatches with the number of"
            f"annotations {len(annotations)}"
        )

        return [
            self.evaluate_sample(texts[i], annotations[i]) for i in range(len(texts))
        ]

    def calculate_score(
        self, evaluation_results: List[EvaluationResult], f_beta: float = 1.0
    ) -> Dict[str, float]:
        # aggregate results
        all_results: Counter = sum(
            [res.label_pair_counter for res in evaluation_results], Counter()
        )

        # compute score per entity
        entity_recall = {}
        entity_precision = {}
        entity_f_score = {}

        for entity in self.target_entities:
            if self.entity_mapping:
                entity = self.entity_mapping[entity]

            annotated = sum(
                [all_results[x] for x in all_results if x.annotated == entity]
            )
            predicted = sum(
                [all_results[x] for x in all_results if x.predicted == entity]
            )
            tp = all_results[(entity, entity)]

            if annotated > 0:
                entity_recall[entity] = tp / annotated
            else:
                entity_recall[entity] = np.NaN

            if predicted > 0:
                per_entity_tp = all_results[(entity, entity)]
                entity_precision[entity] = per_entity_tp / predicted
            else:
                entity_precision[entity] = np.NaN

            if annotated > 0 and predicted > 0:
                entity_f_score[entity] = compute_f_beta(
                    entity_recall[entity], entity_precision[entity], f_beta
                )
            else:
                entity_f_score[entity] = np.NaN

        return entity_f_score
