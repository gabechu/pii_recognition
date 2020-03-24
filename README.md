# Personal Identifiable Information (PII) Recognition

## Quick Start

### Example Usage
#### CRF Model
Let's load a pretrained CRF PII recogniser and run the recogniser over an example text.

```python
from tokeniser.tokeniser import nltk_word_tokenizer
from recognisers.crf_recogniser import CrfRecogniser

crf_recogniser = CrfRecogniser(
    supported_entities=[
        "B-LOC", "I-LOC", "B-ORG", "I-ORG",
        "B-PER", "I-PER", "B-MISC", "I-MISC",
    ],
    supported_languages=["en"],
    model_path="exported_models/conll2003-en.crfsuite",  # pretrained model
    tokenizer=nltk_word_tokenizer,  # this crf is token based
)

crf_recogniser.analyse(text="I love Melbourne.", entities=["I-PER", "I-LOC"])
```

This should print (span is used for segment labelling)
```console
[SpanLabel(entity_type='I-LOC', start=7, end=16)]
```


#### spaCy Model
Create a spaCy recogniser and conduct analysis

```python
from recognisers.spacy_recogniser import SpacyRecogniser

spacy_recogniser = SpacyRecogniser(
    supported_entities=["LOC", "MISC", "ORG", "PER"],
    supported_languages=["en", "de", "es", "fr", "it", "pt", "ru"],
    model_name="xx_ent_wiki_sm"  # more models on https://spacy.io/models
)
spacy_recogniser.analyse(text="I love Melbourne.", entities=["PER", "LOC"])
```

This should also print
```console
[SpanLabel(entity_type='LOC', start=7, end=16)]
```


## Recogniser Evaluation
### Data Format
The *input data* to evaluation is a list of strings, where each string represents either a sentence or a paragraph, for example,
```python
input_data = ["A sentence to be evaluated.", "A paragraph to be evaluated."]
```

The label of *ground truth* is defined at a token-level, that is, assigning an entity label to every token in the text, for example, if using a BIO schema ground truths for the above input data are
```python
ground_truths = [["O", "O", "O", "O", "O", "O"], ["O", "O", "O", "O", "O", "O"]]
```

### Evaluator
Evaluation is based on `f-score`. Take one specific recogniser and pass to the evaluator, depending on the value of `f_beta`, `f1` or `f2` values will be produced for each desried entity.
```python
from evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    recogniser=some_recogniser,
    target_entities=["I-PER"],
    tokeniser=nltk_word_tokenizer  # labels are token based
)

results = evaluator.evaulate_all(input_data, ground_truths)
score = evaluator.calculate_score(results, f_beta=1.)
```
The evaluation produces per entity based results, e.g., `{"PER": 0.7, "LOC": 0.8}`. An aggregation score will be incorporated as an enhancement.

### Performance
Evaluation of experiments are conducted on CONLL 2003 English data -- `eng.testb`. The performance has been logged with `mlflow` and measured by `f1`, `f2` (optional), `precision` and `recall`. I obtained the copy of this CONLL dataset on github, whether the github author posted the full dataset is unknown and whether the author follows the latest guideline generating the CONLL data is also unknown. Further investigation will be taken to decide if we can trust the results getting from this CONLL evaluation. But for now, here's the unvalidated performance


| Experiment | Run | Test Set | Recall | Precision | F1 |  Evaluation Duration |
| -------------    | ------------- |------------- |------------- |------------- |------------- |------------- |
| Heuristic | first_letter_uppercase |  Conll-03 en.testb  |  0.973 | 0.298| 0.456| 1.1s   |
| CRF       | python_crf_no_pos      |  Conll-03 en.testb  |  0.887 | 0.824| 0.854| 1.4s   |
| Spacy     | en_core_web_lg         |  Conll-03 en.testb  |  0.824 | 0.828| 0.826| 6.7s   |
|           | xx_ent_wiki_sm         |  Conll-03 en.testb  |  0.764 | 0.789| 0.776| 6.9s   |
|Flair      | pretrained_en          |  Conll-03 en.testb  |  0.986 | 0.980| 0.983| 32.6min|
|Stanza     | pretrained_en          |  Conll-03 en.testb  |  0.855 | 0.846| 0.850| 10.6min|
