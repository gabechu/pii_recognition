from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.flair import Flair
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import FLAIR
from .mlflow_log import log_evaluation_to_mlflow

RUN_NAME = "ner-en"

recogniser = Flair(
    supported_entities=["PER", "LOC", "ORG", "MISC"],
    supported_languages=["en"],
    model_name="ner",
)


evaluator = ModelEvaluator(
    recogniser, ["PER"], nltk_word_tokenizer, to_eval_labels={"PER": "I-PER"}
)

X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=space_join_detokensier
)

log_evaluation_to_mlflow(FLAIR, RUN_NAME, recogniser, evaluator, X_test, y_test)
