import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# reduce label discrepancies in evaluation metrics when
# running on distinct datasets, e.g., CONLL is using "I-PER" to
# reference person but in WNUT the label is "I-person"
LABEL_COMPLIANCE = {"I-PER": "PERSON", "I-person": "PERSON"}
