from typing import Dict, List, Optional, Set

from pakkr import Pipeline, returns
from pii_recognition.data_readers.data import Data
from pii_recognition.data_readers.presidio_fake_pii_reader import \
    PresidioFakePiiReader
from pii_recognition.evaluation.character_level_evaluation import (
    TextScore, build_label_mapping, compute_entity_precisions_for_prediction,
    compute_entity_recalls_for_ground_truth, compute_pii_detection_f1)
from pii_recognition.recognisers import registry as recogniser_registry
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser
from pii_recognition.utils import dump_to_json_file, load_yaml_file


@returns(Data)
def read_benchmark_data(benchmark_data_file: str) -> Data:
    reader = PresidioFakePiiReader()
    data = reader.build_data(benchmark_data_file)

    # remove empty items
    data.items = list(filter(lambda item: item.text != "", data.items))
    return data


@returns(Data)
def identify_pii_entities(
    data: Data, recogniser_name: str, recogniser_params: Dict
) -> Data:
    recogniser: EntityRecogniser = recogniser_registry.create_instance(
        recogniser_name, recogniser_params
    )

    for item in data.items:
        item.pred_labels = recogniser.analyse(item.text, recogniser.supported_entities)
    return data


@returns(List)
def calculate_precisions_and_recalls(
    data: Data,
    grouped_targeted_labels: List[Set[str]],
    nontargeted_labels: Optional[Set[str]] = None,
) -> List[TextScore]:
    label_mapping = build_label_mapping(grouped_targeted_labels, nontargeted_labels)

    scores = []
    for item in data.items:
        if item.pred_labels:
            pred_labels = item.pred_labels
        else:  # pred_labels could be None
            pred_labels = []

        ent_precisions = compute_entity_precisions_for_prediction(
            len(item.text), item.true_labels, pred_labels, label_mapping
        )
        ent_recalls = compute_entity_recalls_for_ground_truth(
            len(item.text), item.true_labels, pred_labels, label_mapping
        )
        ticket_score = TextScore(precisions=ent_precisions, recalls=ent_recalls)
        scores.append(ticket_score)

    return scores


@returns(Dict)
def calculate_aggregate_metrics(
    scores: List[TextScore], f1_beta: float = 1.0
) -> Dict[str, float]:
    results = dict()
    exact_match = get_rollup_f1s_on_pii(scores, f1_beta, recall_threshold=None)
    results["exact_match_f1"] = sum(exact_match) / len(exact_match)

    partial_match = get_rollup_f1s_on_pii(scores, f1_beta, recall_threshold=0.5)
    results["partial_match_f1_threshold_at_50%"] = sum(partial_match) / len(
        partial_match
    )
    return results


@returns()
def report_results(results: Dict[str, float], dump_file: str):
    dump_to_json_file(results, dump_file)


# TODO: F1 is an accurate name, change to fbeta
def get_rollup_f1s_on_pii(
    scores: List[TextScore], f1_beta: float, recall_threshold: Optional[float]
) -> List[float]:
    f1s = []
    for text_score in scores:
        precisions = [p.precision for p in text_score.precisions]
        recalls = [r.recall for r in text_score.recalls]

        if not precisions and recalls:
            # there are true entities but the system predicts nothing
            f1 = 0.0
        elif precisions and not recalls:
            # there is no true entity but the system predicts something
            f1 = 0.0
        elif not precisions and not recalls:
            # there is no true entity and the system predicts nothing
            f1 = 1.0
        else:
            f1 = compute_pii_detection_f1(
                precisions, recalls, recall_threshold, f1_beta
            )
        f1s.append(f1)
    return f1s


def exec_pipeline(config_yaml_file: str):
    pipeline = Pipeline(
        read_benchmark_data,
        identify_pii_entities,
        calculate_precisions_and_recalls,
        calculate_aggregate_metrics,
        report_results,
        name="pii_validation_pipeline",
    )

    config = load_yaml_file(config_yaml_file)
    if config:
        # conversions to meet requirements on type checks
        config["grouped_targeted_labels"] = [
            set(item) for item in config["grouped_targeted_labels"]
        ]
        config["nontargeted_labels"] = set(config["nontargeted_labels"])
        return pipeline(**config)
    else:
        raise ValueError("Config YAML is empty.")
