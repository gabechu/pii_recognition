from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.crf_recogniser import CrfRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import CRF_EXP, crf_1, crf_2
from .mlflow_log import log_evaluation_to_mlflow

PARAMS = [crf_1, crf_2]
RUN_NAME = "crf_no_pos"

for param in PARAMS:
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
        model_path=param["model_path"],
        tokenizer=param["tokeniser"],
    )
    evaluator = ModelEvaluator(recogniser, ["I-PER"], nltk_word_tokenizer)
    X_test, y_test = get_conll_eval_data(
        file_path=param["eval_data"], detokenizer=space_join_detokensier
    )

    log_evaluation_to_mlflow(
        CRF_EXP, param, recogniser, evaluator, X_test, y_test, run_name=RUN_NAME
    )
