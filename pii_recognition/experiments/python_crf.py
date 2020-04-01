from data_reader.conll_reader import get_conll_eval_data
from data_reader.wnut_reader import get_wnut_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.crf_recogniser import CrfRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import CRF_EXP, crf_1, crf_2, crf_3
from .mlflow_tracking import log_evaluation_to_mlflow

PARAMS = [crf_1, crf_2, crf_3]
RUN_NAME = "crf_no_pos"

for param in PARAMS:
    recogniser = CrfRecogniser(
        supported_entities=param["supported_entities"],
        supported_languages=param["supported_languages"],
        model_path=param["model_path"],
        tokenizer=param["tokeniser"],
    )

    evaluator = None
    X_test = None
    y_test = None

    if "conll2003" in param["eval_data"]:
        X_test, y_test = get_conll_eval_data(
            file_path=param["eval_data"], detokenizer=space_join_detokensier
        )
        evaluator = ModelEvaluator(recogniser, ["I-PER"], nltk_word_tokenizer)

    elif "wnut2017" in param["eval_data"]:
        X_test, y_test = get_wnut_eval_data(
            file_path=param["eval_data"], detokenizer=space_join_detokensier
        )
        evaluator = ModelEvaluator(
            recogniser,
            ["I-PER"],
            nltk_word_tokenizer,
            to_eval_labels={"I-PER": "I-person"},
        )

    if evaluator and X_test and y_test:
        log_evaluation_to_mlflow(
            CRF_EXP, param, recogniser, evaluator, X_test, y_test, run_name=RUN_NAME
        )
