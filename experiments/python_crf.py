from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.crf_recogniser import CrfRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import CRF_EXP
from .mlflow_log import log_evaluation_to_mlflow

RUN_NAME = "word2feature"

recogniser = CrfRecogniser(
    supported_entities=[
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "B-PER",
        "I-PER",
        "B-MISC",
        "I-MISC",
    ],
    supported_languages=["en"],
    model_path="exported_models/conll2003-en.crfsuite",
    tokenizer=nltk_word_tokenizer,
)
evaluator = ModelEvaluator(recogniser, ["I-PER"], nltk_word_tokenizer)

X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=space_join_detokensier
)

log_evaluation_to_mlflow(CRF_EXP, RUN_NAME, recogniser, evaluator, X_test, y_test)
