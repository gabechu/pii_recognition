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
from tqdm import tqdm


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

    for item in tqdm(data.items):
        item.pred_labels = recogniser.analyse(item.text, recogniser.supported_entities)
    return data


@returns(scores=List)
def calculate_precisions_and_recalls(
    data: Data,
    grouped_targeted_labels: List[Set[str]],
    nontargeted_labels: Optional[Set[str]] = None,
) -> Dict[str, List[TextScore]]:
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
        ticket_score = TextScore(
            text=item.text, precisions=ent_precisions, recalls=ent_recalls
        )
        scores.append(ticket_score)

    return {"scores": scores}


@returns()
def log_predictions_and_ground_truths(
    predictions_dump_path: str, scores: List[TextScore]
):
    results = dict()
    for score in scores:
        text = score.text
        predictions = {
            text[p.entity.start : p.entity.end]: {
                "type": p.entity.entity_type,
                "score": round(p.precision, 2),
                "start": p.entity.start,
            }
            for p in score.precisions
        }
        ground_truths = {
            text[r.entity.start : r.entity.end]: {
                "type": r.entity.entity_type,
                "score": round(r.recall, 2),
                "start": r.entity.start,
            }
            for r in score.recalls
        }

        results.update(
            {text: {"predicted": predictions, "ground_truth": ground_truths}}
        )

    dump_to_json_file(results, predictions_dump_path)


@returns(Dict)
def calculate_aggregate_metrics(
    scores: List[TextScore],
    grouped_targeted_labels: List[Set[str]],
    fbeta: float = 1.0,
) -> Dict[Union[str, FrozenSet[str]], float]:
    results: Dict[Union[str, FrozenSet[str]], float] = dict()

    results["exact_match_f1"] = get_rollup_fscore_on_pii(
        scores, fbeta, recall_threshold=None)
    results["partial_match_f1_threshold_at_50%"] = get_rollup_fscore_on_pii(
        scores, fbeta, recall_threshold=0.5
    )

    type_scores: Mapping = get_rollup_metrics_on_types(
        grouped_targeted_labels, scores, fbeta
    )
    results.update(type_scores)

    return results


@returns()
def report_results(results: Dict, scores_dump_path: str):
    results = stringify_keys(results)
    dump_to_json_file(results, scores_dump_path)


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
        return round(sum(fscores) / len(fscores), 4)
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


def regroup_scores_on_types(
    grouped_labels: List[Set[str]], scores: List[TextScore]
) -> Dict[FrozenSet[str], Dict]:
    """Regroup scores according to parameter grouped_labels.

    Prediction scores (precisions and recalls) are collected for each of the example
    texts and stored in scores parameter. We need to regroup those scores on entity
    types to obtain, for example, all precisions and recalls for the group
    {"PER", "PERSON"}.

    Args:
        grouped_llabels: entity labels separated as sets of groups, for example,
            [{"PER", "PERSON"}, {"ORG"}].
        scores: a list of text scores providing info including precisions and recalls
            for each prediction and ground truth within a text.

    Returns:
        A dictionary that key is a group of entities and value precisions and recalls
        for that group.
    """
    score_table: Dict[FrozenSet, Dict] = {
        frozenset(label_set): {"precisions": [], "recalls": []}
        for label_set in grouped_labels
    }

    # update score table
    for text_score in scores:
        for precision in text_score.precisions:
            score_table = _update_table(score_table, precision)
        for recall in text_score.recalls:
            score_table = _update_table(score_table, recall)

    return score_table


def get_rollup_metrics_on_types(
    grouped_labels: List[Set[str]], scores: List[TextScore], fbeta: float,
) -> Dict[FrozenSet[str], Dict[str, Union[float, str]]]:
    """Calculate f1, average precision and average recall for every group in the
    grouped labels.
    """
    score_table = regroup_scores_on_types(grouped_labels, scores)

    metrics = dict()
    for key, value in score_table.items():
        f1 = round(
            compute_pii_detection_fscore(
                value["precisions"], value["recalls"], beta=fbeta
            ),
            4,
        )

        if value["precisions"]:
            ave_precision = round(
                sum(value["precisions"]) / len(value["precisions"]), 4
            )
        else:
            ave_precision = "undefined"

        if value["recalls"]:
            ave_recall = round(sum(value["recalls"]) / len(value["recalls"]), 4)
        else:
            ave_recall = "undefined"

        metrics.update(
            {key: {"f1": f1, "ave-precision": ave_precision, "ave-recall": ave_recall}}
        )
    return metrics


def exec_pipeline(config_yaml_file: str):
    pipeline = Pipeline(
        read_benchmark_data,
        identify_pii_entities,
        calculate_precisions_and_recalls,
        log_predictions_and_ground_truths,
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
