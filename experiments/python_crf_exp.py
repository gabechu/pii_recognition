from evaluation.model_evaluator import ModelEvaluator
from data_reader.conll_reader import get_conll_eval_data
from recognisers.crf_recogniser import CrfRecogniser
from tokeniser.detokeniser import space_join_detokensier
from tokeniser.tokeniser import nltk_word_tokenizer
import mlflow


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


with mlflow.start_run(run_name="python_crf_suite"):
    evaluator = ModelEvaluator(recogniser, ["I-PER"], nltk_word_tokenizer)

    counters, mistakes = evaluator.evaulate_all(X_test, y_test)
    # remove returns with no mistakes
    mistakes = list(filter(lambda x: x.token_errors, mistakes))
    recall, precision, f1 = evaluator.calculate_score(counters, f_beta=1.0)
    _, _, f2 = evaluator.calculate_score(counters, f_beta=2.0)

    mlflow.log_metric("PER_recall", recall["I-PER"])
    mlflow.log_metric("PER_precision", precision["I-PER"])
    mlflow.log_metric("PER_f1", f1["I-PER"])
    mlflow.log_metric("PER_f2", f2["I-PER"])
