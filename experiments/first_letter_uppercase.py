from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.first_letter_uppercase import FirstLetterUppercase
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer
from .mlflow_log import log_evaluation_to_mlflow

from .manage_experiments import FIRST_LETTER_UPPERCASE

RUN_NAME = "use_istitle"

recogniser = FirstLetterUppercase(
    supported_languages=["en"], tokeniser=nltk_word_tokenizer
)

evaluator = ModelEvaluator(
    recogniser, ["PER"], nltk_word_tokenizer, to_eval_labels={"PER": "I-PER"}
)

X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=space_join_detokensier
)

log_evaluation_to_mlflow(
    FIRST_LETTER_UPPERCASE, RUN_NAME, recogniser, evaluator, X_test, y_test
)
