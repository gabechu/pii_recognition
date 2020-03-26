from tokeniser.tokeniser import nltk_word_tokenizer


CRF_EXP = "PythonCRF"
SPACY_EXP = "Spacy"
FIRST_LETTER_UPPERCASE = "FirstLetterUppercase"
FLAIR = "Flair"
STANZA = "Stanza"

crf_1 = {
    "supported_entities": [
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "B-PER",
        "I-PER",
        "B-MISC",
        "I-MISC",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testb",
    "model_path": "exported_models/conll2003-en.crfsuite",
    "tokeniser": nltk_word_tokenizer,
}
crf_2 = {
    "supported_entities": [
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "B-PER",
        "I-PER",
        "B-MISC",
        "I-MISC",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testa",
    "model_path": "exported_models/conll2003-en.crfsuite",
    "tokeniser": nltk_word_tokenizer,
}
crf_3 = {
    "supported_entities": [
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "B-PER",
        "I-PER",
        "B-MISC",
        "I-MISC",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/wnut2017/emerging.test.annotated",
    "model_path": "exported_models/conll2003-en.crfsuite",
    "tokeniser": nltk_word_tokenizer,
}

spacy_1 = {
    "eval_data": "datasets/conll2003/eng.testb",
    "model_name": "en_core_web_lg",
}
spacy_2 = {
    "eval_data": "datasets/wnut2017/emerging.test.annotated",
    "model_name": "en_core_web_lg",
}
