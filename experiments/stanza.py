from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.stanza import Stanza
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import STANZA
from .mlflow_log import log_evaluation_to_mlflow

RUN_NAME = "pretrained_en"

recogniser = Stanza(
    supported_entities=[
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ],
    supported_languages=["en"],
    model_name="en",
)
evaluator = ModelEvaluator(
    recogniser, ["PERSON"], nltk_word_tokenizer, {"PERSON": "I-PER"}
)

X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=space_join_detokensier
)

log_evaluation_to_mlflow(STANZA, RUN_NAME, recogniser, evaluator, X_test, y_test)
