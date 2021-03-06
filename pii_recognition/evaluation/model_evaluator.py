from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from pii_recognition.labels.mapping import map_labels, mask_labels
from pii_recognition.labels.schema import EvalLabel, Entity, TokenLabel
from pii_recognition.labels.span import span_labels_to_token_labels
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser
from pii_recognition.tokenisation.tokenisers import Tokeniser

from .metrics import compute_f_beta
from .prediction_error import SampleError, TokenError


class ModelEvaluator:
    """
    Evaluates a named entity recogniser.

    Attributes:
        recogniser: an instanace of EntityRecogniser, a named entity recogniser.
        tokeniser: an instance of Tokeniser.
        target_entities: entities to be evaluated, using entity labels from the
            recogniser.
        switch_labels: a dict {model_label: test_data_label} facilitates the entity
            conversion between predicted and test labels. Predicted entity labels could
            differ from the test entity labels, e.g., PERSON and PER.
    """

    def __init__(
        self,
        recogniser: EntityRecogniser,
        tokeniser: Tokeniser,
        target_entities: List[str],
        switch_labels: Optional[Dict[str, str]] = None,
    ):
        self.recogniser = recogniser
        self.tokeniser = tokeniser

        # TODO: Add many-to-one support. The switch_labels dict supports one to one and
        # one to many
        self._switch_labels = switch_labels
        self._validate__switch_labels()

        self.target_entities = target_entities
        self._validate_target_entities()

        if switch_labels:
            self._translated_entities = map_labels(target_entities, switch_labels)
        else:
            self._translated_entities = target_entities

    def _validate__switch_labels(self):
        """
        Switch labels is a dict attribute with keys of it are entity names supplied
        in recogniser.
        """
        if self._switch_labels is None:
            pass
        else:
            switch_keys_set = set(self._switch_labels.keys())
            recogniser_entity_set = set(self.recogniser.supported_entities)
            if not (switch_keys_set <= recogniser_entity_set):
                unsupported = switch_keys_set - recogniser_entity_set
                raise ValueError(
                    f"Must use entity names from recogniser for switch_label keys, "
                    f"but contains unknown labels {unsupported}."
                )

    def _validate_target_entities(self):
        """Target entities must using entity labels defined by recogniser."""
        target_entity_set = set(self.target_entities)
        recogniser_entity_set = set(self.recogniser.supported_entities)

        if not (target_entity_set <= recogniser_entity_set):
            unsupported = target_entity_set - recogniser_entity_set
            raise ValueError(
                f"Entities taken for evaluation must use recogniser labels, "
                f"but contains unknown labels {unsupported}."
            )

    def _validate_predictions(self, predicted: List[str]):
        """
        Validate predicted entity labels. Predictions should not contain any unasked
        entities.
        """
        asked_entities = set(self.target_entities) | {"O"}
        predicted_entities = set(predicted)
        assert predicted_entities.issubset(asked_entities), (
            f"Predictions contain unasked entities "
            f"{sorted(list(predicted_entities - asked_entities))}"
        )

    def get_span_based_prediction(self, text: str) -> List[Entity]:
        predicted_spans = self.recogniser.analyse(text, self.target_entities)
        if predicted_spans:
            self._validate_predictions([label.entity_type for label in predicted_spans])
            return predicted_spans
        # TODO: use optional instead of returning empty list
        return []

    def get_token_based_prediction(self, text: str) -> List[TokenLabel]:
        recognised_entities = self.get_span_based_prediction(text)
        tokens = self.tokeniser.tokenise(text)
        token_labels = span_labels_to_token_labels(recognised_entities, tokens)

        return token_labels

    def _compare_predicted_and_truth(
        self,
        text: str,
        tokens: List[str],
        annotations: List[str],
        predictions: List[str],
    ) -> Tuple[Counter, Optional[SampleError]]:
        """
        Comparing the predicted labels (predictions) identified from a given text to
        the labels of ground truth (annotations). A counter and a container that holds
        prediction errors are returned.
        """
        if self._switch_labels:
            predictions = map_labels(predictions, self._switch_labels)

        label_pair_counter: Counter = Counter()
        # token mismatch -- the tokeniser we use produces different token set from the
        # test dataset, cannot perform evaluation.
        if len(annotations) != len(predictions):
            return (
                label_pair_counter,
                SampleError(token_errors=[], full_text=text, failed=True),
            )

        sample_error = SampleError(token_errors=[], full_text=text, failed=False)

        for i in range(len(annotations)):
            label_pair_counter[EvalLabel(annotations[i], predictions[i])] += 1
            # log mistakes
            if annotations[i] != predictions[i]:
                sample_error.token_errors.append(
                    TokenError(
                        annotation=annotations[i],
                        prediction=predictions[i],
                        text=tokens[i],
                    )
                )

        # avoid variable reassignment; otherwise mypy would complain
        if (sample_error.failed is False) and (not sample_error.token_errors):
            rectified_sample_error = None  # if no error found
        else:
            rectified_sample_error = sample_error

        return label_pair_counter, rectified_sample_error

    def evaluate_sample(
        self, text: str, annotations: List[str]
    ) -> Tuple[Counter, Optional[SampleError]]:
        masked_annotations = mask_labels(annotations, self._translated_entities)

        # make prediction
        token_based_predictions = self.get_token_based_prediction(text)
        predictions = [pred.entity_type for pred in token_based_predictions]
        translated_predictions = (
            map_labels(predictions, self._switch_labels)
            if self._switch_labels
            else predictions
        )

        # log results
        tokens = [text[pred.start : pred.end] for pred in token_based_predictions]
        label_pair_counter, sample_error = self._compare_predicted_and_truth(
            text, tokens, masked_annotations, translated_predictions
        )

        return label_pair_counter, sample_error

    def evaluate_all(
        self, texts: List[str], annotations: List[List[str]]
    ) -> Tuple[List[Counter], List[SampleError]]:
        assert len(texts) == len(annotations)

        counters = []
        mistakes = []
        for i in range(len(texts)):
            label_pair_counter, sample_error = self.evaluate_sample(
                texts[i], annotations[i]
            )
            counters.append(label_pair_counter)
            if sample_error is not None:
                mistakes.append(sample_error)
        return counters, mistakes

    def calculate_score(
        self,
        all_eval_counters: List[Counter],
        f_beta: float = 1.0,
        use_test_labels: bool = True,
    ) -> Tuple[Dict, Dict, Dict]:
        # aggregate results
        all_results: Counter = sum(all_eval_counters, Counter())

        # compute score per entity
        entity_recall = {}
        entity_precision = {}
        entity_f_score = {}

        for entity in self._translated_entities:
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

        # use recogniser entity labels
        if (use_test_labels is False) and (self._switch_labels is not None):
            convert_to_recogniser_labels = {
                value: key for key, value in self._switch_labels.items()
            }

            entity_recall = self._convert_metric_labels(
                entity_recall, convert_to_recogniser_labels
            )
            entity_precision = self._convert_metric_labels(
                entity_precision, convert_to_recogniser_labels
            )
            entity_f_score = self._convert_metric_labels(
                entity_f_score, convert_to_recogniser_labels
            )

        return entity_recall, entity_precision, entity_f_score

    def _convert_metric_labels(
        self, metric: Dict[str, float], converter: Dict[str, str]
    ) -> Dict[str, float]:
        return {converter[name]: score for name, score in metric.items()}
