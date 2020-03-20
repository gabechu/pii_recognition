from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.spacy_recogniser import SpacyRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import Spacy_EXP
from .mlflow_log import log_evaluation_to_mlflow

RUN_NAME = "xx_ent_wiki_sm"

recogniser = SpacyRecogniser(
    supported_entities=["LOC", "MISC", "ORG", "PER"],
    supported_languages=["en", "de", "es", "fr", "it", "pt", "ru"],
    model_name="xx_ent_wiki_sm",
)

evaluator = ModelEvaluator(
    recogniser, ["PER"], nltk_word_tokenizer, to_eval_labels={"PER": "I-PER"}
)

X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=space_join_detokensier
)

log_evaluation_to_mlflow(Spacy_EXP, RUN_NAME, recogniser, evaluator, X_test, y_test)
