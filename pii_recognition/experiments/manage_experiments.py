from tokeniser.tokeniser import nltk_word_tokenizer


CRF_EXP = "PythonCRF"
SPACY = "Spacy"
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

spacy_spacy_en_core_web_lg_1 = {
    "supported_entities": [
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testb",
    "model_name": "en_core_web_lg",
}
spacy_spacy_en_core_web_lg_2 = {
    "supported_entities": [
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testa",
    "model_name": "en_core_web_lg",
}
spacy_spacy_en_core_web_lg_3 = {
    "supported_entities": [
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/wnut2017/emerging.test.annotated",
    "model_name": "en_core_web_lg",
}

spacy_xx_ent_wiki_sm_1 = {
    "supported_entities": ["LOC", "MISC", "ORG", "PER"],
    "supported_languages": ["en", "de", "es", "fr", "it", "pt", "ru"],
    "eval_data": "datasets/conll2003/eng.testb",
    "model_name": "xx_ent_wiki_sm",
}
spacy_xx_ent_wiki_sm_2 = {
    "supported_entities": ["LOC", "MISC", "ORG", "PER"],
    "supported_languages": ["en", "de", "es", "fr", "it", "pt", "ru"],
    "eval_data": "datasets/conll2003/eng.testa",
    "model_name": "xx_ent_wiki_sm",
}
spacy_xx_ent_wiki_sm_3 = {
    "supported_entities": ["LOC", "MISC", "ORG", "PER"],
    "supported_languages": ["en", "de", "es", "fr", "it", "pt", "ru"],
    "eval_data": "datasets/wnut2017/emerging.test.annotated",
    "model_name": "xx_ent_wiki_sm",
}

stanza_1 = {
    "supported_entities": [
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testb",
    "model_name": "en",
}
stanza_2 = {
    "supported_entities": [
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testa",
    "model_name": "en",
}
stanza_3 = {
    "supported_entities": [
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ],
    "supported_languages": ["en"],
    "eval_data": "datasets/wnut2017/emerging.test.annotated",
    "model_name": "en",
}

flair_1 = {
    "supported_entities": ["PER", "LOC", "ORG", "MISC"],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testb",
    "model_name": "ner",
}
flair_2 = {
    "supported_entities": ["PER", "LOC", "ORG", "MISC"],
    "supported_languages": ["en"],
    "eval_data": "datasets/conll2003/eng.testa",
    "model_name": "ner",
}
flair_3 = {
    "supported_entities": ["PER", "LOC", "ORG", "MISC"],
    "supported_languages": ["en"],
    "eval_data": "datasets/wnut2017/emerging.test.annotated",
    "model_name": "ner",
}

first_letter_uppercase_3 = {
    "supported_entities": ["PER"],
    "supported_languages": ["en"],
    "eval_data": "datasets/wnut2017/emerging.test.annotated",
    "tokeniser": nltk_word_tokenizer,
}
