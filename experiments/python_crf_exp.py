import os
from typing import List

import mlflow

from data_reader.conll_reader import get_conll_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.crf_recogniser import CrfRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer
from utils import write_iterable_to_text

from .manage_experiments import CRF_EXP

RUN_NAME = "word2feature"


def write_plain_text(content: List, file_path: str):
    with open(file_path, "w") as f:
        for elem in content:
            f.write(str(elem) + "\n")


# init recogniser
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


# Prepare evalution data
X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=space_join_detokensier
)


# Log to mlflow
mlflow.set_experiment(CRF_EXP)
with mlflow.start_run(run_name=RUN_NAME):
    evaluator = ModelEvaluator(recogniser, ["I-PER"], nltk_word_tokenizer)

    counters, mistakes = evaluator.evaulate_all(X_test, y_test)
    # remove returns with no mistakes
    mistakes = list(filter(lambda x: x.token_errors, mistakes))
    recall, precision, f1 = evaluator.calculate_score(counters, f_beta=1.0)
    _, _, f2 = evaluator.calculate_score(counters, f_beta=2.0)

    write_iterable_to_text(mistakes, f"{RUN_NAME}.mis")
    mlflow.log_artifact(f"{RUN_NAME}.mis")

    os.remove(f"{RUN_NAME}.mis")

    mlflow.log_metric("PER_recall", recall["I-PER"])
    mlflow.log_metric("PER_precision", precision["I-PER"])
    mlflow.log_metric("PER_f1", f1["I-PER"])
    mlflow.log_metric("PER_f2", f2["I-PER"])
