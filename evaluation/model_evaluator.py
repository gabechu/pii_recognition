from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from label.label_schema import EvalLabel
from label.mapping import map_labels
from label.span_to_token import span_labels_to_token_labels
from recognisers.entity_recogniser import Rec_co
from tokeniser.token import Token

from .metrics import compute_f_beta
from .prediction_error import SampleError, TokenError


class ModelEvaluator:
    """
    Evaluates a named entity recogniser.

    Attributes:
        recogniser: a named entity recogniser.
        target_entities: entities to be evaluated.
        tokeniser: a callable to break a string into tokens.
        entity_mapping: a dict facilitate entity conversion. Predicted entity labels
            may differ from evaluation entity labels, e.g., PERSON and PER.
    """

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

    def get_token_based_prediction(self, text: str) -> List[str]:
        recognised_entities = self.recogniser.analyse(text, self.target_entities)
        tokens = self.tokeniser(text)
        token_labels = span_labels_to_token_labels(recognised_entities, tokens)

        # validate predictions
        asked_entities = set(self.target_entities) | {"O"}
        predicted_entities = set(token_labels)
        assert predicted_entities.issubset(asked_entities), (
            f"Predictions contain unasked entities "
            f"{sorted(list(predicted_entities - asked_entities))}"
        )

        return token_labels

    def _compare_predicted_and_truth(
        self, text: str, annotations: List[str], predictions: List[str]
    ) -> Tuple[Counter, SampleError]:
        """
        Given a sample text, compare the ground truth entity labels dennoted by
        annotation and predicted entity labels denoted by predictions. Count the
        occurrence and find mistakes.
        """
        if self.entity_mapping:
            predictions = map_labels(predictions, self.entity_mapping)

        label_pair_counter = Counter()
        if len(annotations) != len(predictions):
            return (
                label_pair_counter,
                SampleError(token_errors=[], full_text=text, failed=True),
            )

        sample_error = SampleError(token_errors=[], full_text=text, failed=False)
        tokens = self.tokeniser(text)

        for i in range(len(annotations)):
            label_pair_counter[EvalLabel(annotations[i], predictions[i])] += 1
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

    def evaluate_sample(
        self, text: str, annotations: List[str]
    ) -> Tuple[Counter, SampleError]:
        # TODO: filter annotation by target_entity
        predictions = self.get_token_based_prediction(text)
        label_pair_counter, sample_error = self._compare_predicted_and_truth(
            text, annotations, predictions
        )
        return label_pair_counter, sample_error

    def evaulate_all(
        self, texts: List[str], annotations: List[List[str]]
    ) -> Tuple[List[Counter], List[SampleError]]:
        assert len(texts) == len(annotations), (
            f"The number of texts: {len(texts)} mismatches with the number of"
            f"annotations {len(annotations)}"
        )

        counters = []
        mistakes = []
        for i in range(len(texts)):
            label_pair_counter, sample_error = self.evaluate_sample(
                texts[i], annotations[i]
            )
            counters.append(label_pair_counter)
            mistakes.append(sample_error)

        return counters, mistakes

    def calculate_score(
        self, all_eval_counters: List[Counter], f_beta: float = 1.0
    ) -> Dict[str, float]:
        # aggregate results
        all_results: Counter = sum(all_eval_counters, Counter())

        # compute score per entity
        entity_recall = {}
        entity_precision = {}
        entity_f_score = {}

        translated_target_entities = [
            self.entity_mapping[entity] if self.entity_mapping else entity
            for entity in self.target_entities
        ]

        for entity in translated_target_entities:
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
