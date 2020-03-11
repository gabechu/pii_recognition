from evaluation.model_evaluator import ModelEvaluator
from data_reader.conll_reader import get_conll_eval_data
from recognisers.crf_recogniser import CrfRecogniser
from recognisers.spacy_recogniser import SpacyRecogniser
from tokenizers.detokenizer import spacy_join_detokenzier
from tokenizers.nltk_tokenizer import word_tokenizer

# Prepare evalution data
X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=spacy_join_detokenzier
)


crf_recogniser = CrfRecogniser(
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
    tokenizer=word_tokenizer,
)
evaluator = ModelEvaluator(crf_recogniser, ["I-PER"], word_tokenizer)


results = evaluator.evaulate_all(X_test, y_test)
score = evaluator.calculate_score(results)
print(score)
