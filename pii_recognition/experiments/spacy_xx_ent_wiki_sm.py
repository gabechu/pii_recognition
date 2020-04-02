from data_reader.conll_reader import get_conll_eval_data
from data_reader.wnut_reader import get_wnut_eval_data
from evaluation.model_evaluator import ModelEvaluator
from recognisers.spacy_recogniser import SpacyRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer

from .manage_experiments import (
    SPACY,
    spacy_xx_ent_wiki_sm_1,
    spacy_xx_ent_wiki_sm_2,
    spacy_xx_ent_wiki_sm_3,
)
from .mlflow_tracking import log_evaluation_to_mlflow

RUN_NAME = "xx_ent_wiki_sm"

PARAMS = [spacy_xx_ent_wiki_sm_1, spacy_xx_ent_wiki_sm_2, spacy_xx_ent_wiki_sm_3]

for param in PARAMS:
    recogniser = SpacyRecogniser(
        supported_entities=param["supported_entities"],
        supported_languages=param["supported_languages"],
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
            recogniser, ["PER"], nltk_word_tokenizer, to_eval_labels={"PER": "I-PER"},
        )
    elif "wnut2017" in param["eval_data"]:
        X_test, y_test = get_wnut_eval_data(
            file_path=param["eval_data"], detokenizer=space_join_detokensier
        )
        evaluator = ModelEvaluator(
            recogniser,
            ["PER"],
            nltk_word_tokenizer,
            to_eval_labels={"PER": "I-person"},
        )

    if evaluator and X_test and y_test:
        log_evaluation_to_mlflow(
            SPACY, recogniser, evaluator, X_test, y_test, run_name=RUN_NAME
        )
