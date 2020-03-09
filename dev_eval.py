from eval.evaluation import bio_classification_report, get_predicted_entities
from reader.conll_reader import get_conll_eval_data, spacy_join_detokenzier
from recognisers.spacy_recogniser import SpacyRecogniser
from recognisers.crf_recogniser import CrfRecogniser
from tokenizers.nltk_tokenizer import word_tokenizer
import mlflow


X_test, y_test = get_conll_eval_data(
    file_path="datasets/conll2003/eng.testb", detokenizer=spacy_join_detokenzier
)

# multilingual_recogniser = SpacyRecogniser(
#     model_name="xx_ent_wiki_sm",
#     supported_entities=["LOC", "MISC", "ORG", "PER"],
#     supported_languages=["en", "es", "fr", "it", "pt", "ru"],
# )
# y_pred = [
#     get_predicted_entities(text, ["PER"], multilingual_recogniser, {"PER": "I-PER"})
#     for text in X_test
# ]


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
y_pred = [
    get_predicted_entities(text, ["B-PER", "I-PER"], crf_recogniser) for text in X_test
]

okay_index = []
counter = 0
for i in range(len(y_pred)):
    if len(y_pred[i]) != len(y_test[i]):
        counter += 1
    else:
        okay_index.append(i)


def select_by_index(target, indices):
    results = []
    for i in range(len(target)):
        if i in indices:
            results.append(target[i])
    return results


print(counter, counter / len(y_pred))
res = bio_classification_report(
    select_by_index(y_test, okay_index), select_by_index(y_pred, okay_index)
)

# res = res['I-PER']['f1_score']
# import pdb; pdb.set_trace()
mlflow.log_metric("I-PER f1-score", res["I-PER"]["f1-score"])
