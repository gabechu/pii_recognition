# Personal Identifiable Information (PII) Recognition

## Quick Start

### Example Usage
Let's run a pretrained CRF PII recogniser over an example text.
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
    tokenizer=nltk_word_tokenizer,
)

crf_recogniser.analyze(text="I love Melbourne.", entities=["I-PER", "I-LOC"])
```

This should print
```console
[RecogniserResult(entity_type='I-LOC', start=7, end=16)]
```


## Recogniser Evaluation
Evaluate one specific recogniser for `f-score`, depending on the value of `f_beta` it can be `f1` or `f2`. 
```python
from evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    recogniser=crf_recogniser,
    target_entities=["I-PER"],
    tokeniser=nltk_word_tokenizer  # labels are token based
)

results = evaluator.evaulate_all(X_test, y_test)
score = evaluator.calculate_score(results, f_beta=1.)
```
Aggregation score could be an enhancement that we'd like to incorporate in the near future.