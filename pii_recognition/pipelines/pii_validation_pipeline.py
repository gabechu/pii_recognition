from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Union

from pakkr import Pipeline, returns
from pii_recognition.data_readers.data import Data
from pii_recognition.data_readers.presidio_fake_pii_reader import PresidioFakePiiReader
from pii_recognition.evaluation.character_level_evaluation import (
    EntityPrecision,
    EntityRecall,
    TextScore,
    build_label_mapping,
    compute_entity_precisions_for_prediction,
    compute_entity_recalls_for_ground_truth,
    compute_pii_detection_fscore,
)
from pii_recognition.recognisers import registry as recogniser_registry
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser
from pii_recognition.utils import dump_to_json_file, load_yaml_file, stringify_keys


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
    scores: List[TextScore],
    grouped_targeted_labels: List[Set[str]],
    fbeta: float = 1.0,
) -> Dict[Union[str, FrozenSet[str]], float]:
    round_ndigits = 4
    results: Dict[Union[str, FrozenSet[str]], float] = dict()

    results["exact_match_f1"] = round(
        get_rollup_fscore_on_pii(scores, fbeta, recall_threshold=None), round_ndigits
    )

    results["partial_match_f1_threshold_at_50%"] = round(
        get_rollup_fscore_on_pii(scores, fbeta, recall_threshold=0.5), round_ndigits
    )

    type_scores: Mapping = get_rollup_fscores_on_types(
        grouped_targeted_labels, scores, fbeta
    )
    type_scores = {
        key: round(value, round_ndigits) for key, value in type_scores.items()
    }
    results.update(type_scores)

    return results


@returns()
def report_results(results: Dict, dump_file: str):
    results = stringify_keys(results)
    dump_to_json_file(results, dump_file)


def get_rollup_fscore_on_pii(
    scores: List[TextScore], fbeta: float, recall_threshold: Optional[float]
) -> float:
    """Calculate f score on PII recognition.

    A single score, f score, will be calculate to indicate how a system did on
    predicting PII entities. Recall thresholding is supported, if the system can
    recognise a certain portion of an entity greater than the threshold, that
    entity then will be considered identified.

    Args:
        scores: a list of text scores providing info including precisions and recalls.
        fbeta: beta value for f score.
        recall_threshold: a float between 0 and 1. Any recall value that is greater
            than or equals to the threshold would be rounded up to 1.

    Returns:
        A f score represents performance of a system.
    """
    fscores = []
    for text_score in scores:
        precisions = [p.precision for p in text_score.precisions]
        recalls = [r.recall for r in text_score.recalls]
        f = compute_pii_detection_fscore(precisions, recalls, recall_threshold, fbeta)
        fscores.append(f)

    if fscores:
        return sum(fscores) / len(fscores)
    else:
        # The only possibility to have empty fscores is that argument "scores"
        # is empty. In this case, we assign f score to 0.
        return 0.0


def _update_table(
    table: Dict[FrozenSet[str], Dict], new_item: Union[EntityPrecision, EntityRecall]
) -> Dict[FrozenSet, Dict]:
    """A helper function to log fscores."""
    entity_label = new_item.entity.entity_type
    for label_set in table.keys():
        if entity_label in label_set:
            if isinstance(new_item, EntityPrecision):
                table[label_set]["precisions"].append(new_item.precision)
            elif isinstance(new_item, EntityRecall):
                table[label_set]["recalls"].append(new_item.recall)
    return table


def get_rollup_fscores_on_types(
    grouped_labels: List[Set[str]], scores: List[TextScore], fbeta: float,
) -> Dict[FrozenSet[str], float]:
    """Calculate a f scores for every group in the grouped labels.

    There are entity labels being grouped and passed to this function as an argument,
    with which we can reorganise the list of scores and calculate f scores accordingly.
    Once it done, we will get a f score for every group in grouped labels.

    Args:
        grouped_llabels: entity labels we are interested that have been
            separated as sets of groups, for example, [{"PER", "PERSON"}, {"ORG"}].
        scores: a list of text scores providing info including precisions and recalls.
        fbeta: beta value for f score.

    Returns:
        A dictionary that key is a group of entities and value is f score for the group.
    """
    score_table: Dict[FrozenSet, Dict] = {
        frozenset(label_set): {"precisions": [], "recalls": [], "f1": None}
        for label_set in grouped_labels
    }

    # update score table
    for text_score in scores:
        for precision in text_score.precisions:
            score_table = _update_table(score_table, precision)
        for recall in text_score.recalls:
            score_table = _update_table(score_table, recall)

    # average precisions and recalls
    for key, value in score_table.items():
        value["f1"] = compute_pii_detection_fscore(
            value["precisions"], value["recalls"], beta=fbeta
        )
    return {key: value["f1"] for key, value in score_table.items()}


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
