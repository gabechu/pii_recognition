# Personal Identifiable Information (PII) Recognition

## Development Progress
This project is still under development and we have finshed the first two stages.

The first stage is to build a benchmark model. We have chosen `CRF` for benchmarking, it is simple and fast. `CRF` has made reasonable assumptions just to solve the kind of problem -- sequence labelling problem, and it is the stepping stone to much complex models. Quick inference which is the biggest advantage with `CRF`. So far `CRF` has been the fastest model we tested at inference time. Feature engineering is where the difficulty lies in developing a high accuracy `CRF` model. This motivates us to look at SOTA models to gain understanding about the performance gaps.

The second stage is to evaluate as many as off-the-shelf NER models and find the most promising one/ones if any. We have used models developed for NER tasks, solving NER is equivalent to solving PII once entities consense. Model evaluation focuses on the conventional metrics `recall`, `precision`, `f1` and time uses for inference as well. Depending on the task, there is no single best model. Complex NER models such as `Flair` does achieve very good accuracies across many of its supported entities, but inference is very very slow if GPU and batch both are disabled. `Spacy` models have shown a good balance -- decent accuracies with quick inference. But the downside of all of the off-the-shelf models is lacking the ability to be extended on new entities unless retraining the model. Besides, definitions of entities are not crystal clear. The same entity name could interpreted differently depending on which model you have chosen. For example, `LOC` in Spacy `xx_ent_wiki_sm` model means `LOC` and `GPE` in Spacy `en_core_web_lg` model.

Moving forward we will focus on regex, visualisation and online training. Regex is an enhancement of building bespoke component handling particular entities, for example, medicare number in Australia. Visualisation will an interactive demo showing what's it like on an end-user's perspective. Further down the road, we will enable feedback and collect it feed to the model for online learning where the model can continue improving without us creating new rules.

## Installation
The project is developed with Python3.7, make sure you it available. Other versions of Python may work, but you may have to downgrade specific libraries to fix compatibility issues.

Installing `poetry`, a tool that gracefully handles dependencies for you.
```
pip install poetry
```
Using `install` command to download and install required dependencies listed in `poetry.lock`. This could take a while.
```
poetry install
```

If you want to create virtualenv inside the project's root directory, you can update `poetry` config.
```
poetry config virtualenvs.in-project true
```
Starting a shell and you are ready
```
poetry shell
```

## Quick Start

### Example Usage
#### CRF Model
Load a pretrained CRF model and fire up the analyser.

```python
from pii_recognition.recognisers.crf_recogniser import CrfRecogniser

crf_recogniser = CrfRecogniser(
    supported_entities=["I-LOC", "I-ORG", "I-PER", "I-MISC"],
    supported_languages=["en"],
    model_path="pii_recognition/exported_models/conll2003-en.crfsuite",
    tokeniser_setup={"name": "TreebankWordTokeniser"},
)

crf_recogniser.analyse(text="I love Melbourne.", entities=["I-PER", "I-LOC"])
```

You will get span labels as follows
```console
[SpanLabel(entity_type='I-LOC', start=7, end=16)]
```


#### spaCy Model
Create a spaCy recogniser and kick off the analyser.

```python
from pii_recognition.recognisers.spacy_recogniser import SpacyRecogniser

spacy_recogniser = SpacyRecogniser(
    supported_entities=["LOC", "MISC", "ORG", "PER"],
    supported_languages=["en", "de", "es", "fr", "it", "pt", "ru"],
    model_name="xx_ent_wiki_sm",
)
spacy_recogniser.analyse(text="I love Melbourne.", entities=["PER", "LOC"])
```

You will get span labels as follows
```console
[SpanLabel(entity_type='LOC', start=7, end=16)]
```

#### Other available models
Many other off-the-shelf models are provided as well with the detail implementations found in `recognisers` folder, including two neural networks based inference models [`flair`](https://github.com/flairNLP/flair) and [`stanza`](https://github.com/stanfordnlp/stanza).


#### Customise a Recogniser
Add a custom recogniser by inheriting from `EntityRecogniser` class and implementing `analyse` method.
```python
from typing import List
from pii_recognition.labels.schema import SpanLabel
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser

class CustomRecogniser(EntityRecogniser):
    def __init__(self, supported_entities: List[str], supported_languages: List[str], name: str, **kwargs):
        ...

    def analyse(self, text: str, entities: List[str]) -> List[SpanLabel]:
        ...
```


## Train a Recogniser
Training is not the focus for this project. Two directories, `features` and `exported_models`, have been maintained for training as it is needed for developing CRF models and should be aware that files within are not tested.

## Evaluate a Recogniser
### Evaluation Dataset Format
Evaluation requires sentences to be the *input*.
```python
input_data: List[str] = ["A sentence to be evaluated.", "I love Melbourne."]
```

*Ground truth* is assigned at a token-level for each input sentence. Each token of the sentence will be assigned with an entity label. The label could be in either BIO schema or IO schema.
```python
ground_truths = [["O", "O", "O", "O", "O", "O"], ["O", "O", "I-LOC", "O"]]
```

### Evaluator
Evaluation is measured by `recall`, `precision` and `f-score`. Evaluator takes a recogniser as an input and evaluate it over a provided dataset.

### Pakkr + MLflow Pipeline
Pakkr is a lightweight tool developed by Zendesk for building machine learning pipelines. MLflow Tracking is an API to log parameters and artefacts in machine learning experiments.

Experiments configurations are resided in `pii_recognition/experiments/` in `yaml` format, following `boilerplate.yaml` to create new experiment runs. Available pipelines are resided in `pii_recognition/pipelines` with CLI enabled.

Execute a pipeline via CLI by passing a config yaml file. Be aware batch and GPU are not supported yet, evaluate on deep learning models are slow.
```
python pii_recognition/pipelines/pakkr_evaluation.py
--config_yaml pii_recognition/experiments/you_pick
```

MLflow Tracking will log the run and the associated artefacts to local file or a designated database. Examine the results with an interaction UI available at http://localhost:5000. Start it with:
```
mlflow ui
```

### Performance
Evaluation of experiments are performed on CONLL 2003 English data -- `eng.testb` with `MLflow` on `f1`, `precision` and `recall`. We will be updating the table as the project moves forward.


| Experiment | Run | Test Set | Recall | Precision | F1 |  Evaluation Duration |
| -------------    | ------------- |------------- |------------- |------------- |------------- |------------- |
| Heuristic | first_letter_uppercase |  Conll-03 en.testb  |  0.973 | 0.298| 0.456| 1.1s   |
| CRF       | python_crf_no_pos      |  Conll-03 en.testb  |  0.887 | 0.824| 0.854| 1.4s   |
| Spacy     | en_core_web_lg         |  Conll-03 en.testb  |  0.824 | 0.828| 0.826| 6.7s   |
|           | xx_ent_wiki_sm         |  Conll-03 en.testb  |  0.764 | 0.789| 0.776| 6.9s   |
|Flair      | pretrained_en          |  Conll-03 en.testb  |  0.986 | 0.980| 0.983| 32.6min|
|Stanza     | pretrained_en          |  Conll-03 en.testb  |  0.855 | 0.846| 0.850| 10.6min|

notes: no batch is enabled, all models are running on CPU and tested with English dataset on PERSON entity
