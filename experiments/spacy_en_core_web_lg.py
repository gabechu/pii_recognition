# TODO: adapt changes of mlflow log
from data_reader.wnut_reader import get_wnut_eval_data
from experiments.python_crf import PARAMS
from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.spacy_recogniser import SpacyRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import SPACY_EXP, spacy_1, spacy_2
from .mlflow_log import log_evaluation_to_mlflow

PARAMS = [spacy_1, spacy_2]
RUN_NAME = "spacy_en_core_web_lg"

for param in PARAMS:
    recogniser = SpacyRecogniser(
        supported_entities=[
            "CARDINAL",
            "DATE",
            "EVENT",
            "FAC",
            "GPE",
            "LANGUAGE",
            "LAW",
            "LOC",
            "MONEY",
            "NORP",
            "ORDINAL",
            "ORG",
            "PERCENT",
            "PERSON",
            "PRODUCT",
            "QUANTITY",
            "TIME",
            "WORK_OF_ART",
        ],
        supported_languages=["en"],
        model_name=param["model_name"],
    )

    evaluator = None
    X_test = None
    y_test = None

    if "conll2003" in param["eval_data"]:
        X_test, y_test = get_conll_eval_data(
            file_path=param["eval_data"], detokenizer=space_join_detokensier
        )
        evaluator = ModelEvaluator(
            recogniser, ["PERSON"], nltk_word_tokenizer, to_eval_labels={"PERSON": "I-PER"}
        )
    elif "wnut2017" in param["eval_data"]:
        X_test, y_test = get_wnut_eval_data(
            file_path=param["eval_data"], detokenizer=space_join_detokensier
        )
        evaluator = ModelEvaluator(
            recogniser, ["PERSON"], nltk_word_tokenizer, to_eval_labels={"PERSON": "I-person"}
        )

    if evaluator and X_test and y_test:
        log_evaluation_to_mlflow(
            SPACY_EXP, param, recogniser, evaluator, X_test, y_test, run_name=RUN_NAME
        )
