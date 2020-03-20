from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.spacy_recogniser import SpacyRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import Spacy_EXP
from .mlflow_log import log_evaluation_to_mlflow

RUN_NAME = "spacy_en_core_web_lg"


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
    model_name="en_core_web_lg",
)

evaluator = ModelEvaluator(
        recogniser, ["PERSON"], nltk_word_tokenizer, to_eval_labels={"PERSON": "I-PER"}
)

X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=space_join_detokensier
)

log_evaluation_to_mlflow(Spacy_EXP, RUN_NAME, recogniser, evaluator, X_test, y_test)
