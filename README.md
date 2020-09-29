# Personal Identifiable Information (PII) Recognition

## Development Progress
This project is still under development and we have finshed the first two stages.

The first stage is to build a benchmark model. We have chosen `CRF` for benchmarking, it is simple and fast. `CRF` has made reasonable assumptions just to solve the kind of problem -- sequence labelling problem, and it is the stepping stone to much complex models. Quick inference which is the biggest advantage with `CRF`. So far `CRF` has been the fastest model we tested at inference time. Feature engineering is where the difficulty lies in developing a high accuracy `CRF` model. This motivates us to look at SOTA models to gain understanding about the performance gaps.

The second stage is to evaluate as many as off-the-shelf NER models and find the most promising one/ones if any. We have used models developed for NER tasks, solving NER is equivalent to solving PII once entities consense. Model evaluation focuses on the conventional metrics `recall`, `precision`, `f1` and time uses for inference as well. Depending on the task, there is no single best model. Complex NER models such as `Flair` does achieve very good accuracies across many of its supported entities, but inference is very very slow if GPU and batch both are disabled. `Spacy` models have shown a good balance -- decent accuracies with quick inference. But the downside of all of the off-the-shelf models is lacking the ability to be extended on new entities unless retraining the model. Besides, definitions of entities are not crystal clear. The same entity name could interpreted differently depending on which model you have chosen. For example, `LOC` in Spacy `xx_ent_wiki_sm` model means `LOC` and `GPE` in Spacy `en_core_web_lg` model.

Moving forward we will focus on regex, visualisation and online training. Regex is an enhancement of building bespoke component handling particular entities, for example, medicare number in Australia. Visualisation will an interactive demo showing what's it like on an end-user's perspective. Further down the road, we will enable feedback and collect it feed to the model for online learning where the model can continue improving without us creating new rules.

## Installation
The project is developed with Python3.7, make sure you have it available. We'd recommond to use [`pyenv`](https://github.com/pyenv/pyenv) to switch between multiple Python versions, but be aware dependencies may not resolve successfully, in that case, you will have to fix it manually.

Install `poetry`, a dependencies management tool.
```
pip install poetry
```
Use `install` command of `poetry` to download and install dependencies listed in `poetry.lock`. This may take a while.
```
poetry install
```

Update `poetry` config to create virtualenv inside the project's root directory.
```
poetry config virtualenvs.in-project true
```
Start a shell and you are ready
```
poetry shell
```

## Quick Start

### Example Usage
#### CRF Model
Load a pretrained CRF model and kick off the CRF analyser.

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

You will get predicted labels in spans as follows
```console
[Entity(entity_type='I-LOC', start=7, end=16)]
```


#### spaCy Model
Similarly, create a spaCy recogniser and kick off the Spacy analyser.

```python
from pii_recognition.recognisers.spacy_recogniser import SpacyRecogniser

spacy_recogniser = SpacyRecogniser(
    supported_entities=["LOC", "MISC", "ORG", "PER"],
    supported_languages=["en", "de", "es", "fr", "it", "pt", "ru"],
    model_name="xx_ent_wiki_sm",
)
spacy_recogniser.analyse(text="I love Melbourne.", entities=["PER", "LOC"])
```

You will get predicted labels in span as follows. The results may differ with other example sentences since models beneath of two recognisers are different.
```console
[Entity(entity_type='LOC', start=7, end=16)]
```

#### Other available models
Many other off-the-shelf models are provided, with implementations residing in `pii_recognition/recognisers` folder, for example, one SOTA model built in [`flair`](https://github.com/flairNLP/flair) and another is the popular NLP library developed by Stanford [`stanza`](https://github.com/stanfordnlp/stanza).


#### Customise a Recogniser
Add a custom recogniser by inheriting from `EntityRecogniser` class and implementing `analyse` method.
```python
from typing import List
from pii_recognition.labels.schema import Entity
from pii_recognition.recognisers.entity_recogniser import EntityRecogniser

class CustomRecogniser(EntityRecogniser):
    def __init__(self, supported_entities: List[str], supported_languages: List[str], name: str, **kwargs):
        ...

    def analyse(self, text: str, entities: List[str]) -> List[Entity]:
        ...
```
Then calling `analyse` method for predictions.
```
custom_recogniser = CustomRecogniser(supported_entities, supported_languages, name, **kwargs)
custom_recogniser.analyse(text="I love Melbourne.", entities)
```


## Train a Recogniser
Training will not be the focus of this project until we start online training, you will see no test created for training related files.

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
![Demo Animation](../assets/mlflow_tracking_ui.png?raw=true)

### Performance
Evaluations are carried out on CONLL 2003 and WNUT 2017 datasets and test performance is measured by `f1`, `precision`, `recall` and time of inference, which can be found at `gchu/upload_exp_results` within folder `mlrun`. The following test sets are used:
* `eng.testa` of CONLL 2003
* `eng.testb` of CONLL 2003
* `emerging.test.annotated` of WNUT 2017

----------
Table 1: Evaluation results on `eng.testb` for CONLL 2003

|Experiment |Run             |Num of Examples |I-LOC F1 |I-LOC Precision |I-LOC Recall |I-ORG F1 |I-ORG Precision |I-ORG Recall |I-PER F1 |I-PER Precision |I-PER Recall |Evaluation Duration |
| --------- | -------------- | -------------- | ------- | -------------- | ----------- | ------- | -------------- | ----------- | ------- | -------------- | ----------- | ------------------ |
| Heuristic | Uppercase      | 3453           |         |                |             |         |                |             | 0.4563  | 0.2980         | 0.9732      | 1.6s               |
| CRF       | In-house CRF   | 3453           | 0.7957  | 0.8389         | 0.7567      | 0.7405  | 0.7621         | 0.7200      | 0.8543  | 0.8237         | 0.8872      | 2.0s               |
| Spacy     | en_core_web_lg | 3453           | 0.7670  | 0.7067         | 0.8385      | 0.5734  | 0.5821	      | 0.5650      | 0.8256  | 0.8277         | 0.8235      | 15.8s              |
|           | xx_ent_wiki_sm | 3453           | 0.5966  | 0.5310         | 0.6807      | 0.4483  | 0.6008         | 0.3576	    | 0.7759  | 0.7884         | 0.7639      | 7.4s               |
|Flair      | pretrained_en  | 3453           | 0.7269  | 0.7622         | 0.6947      | 0.8208  | 0.7573         |	0.8960      | 0.8349  | 0.7453         | 0.9490      | 21min              |
|Stanza     | pretrained_en  | 3453           | 0.7874  | 0.7666         | 0.8093      | 0.5206  | 0.6337         | 0.4418      | 0.8488  | 0.8451         | 0.8524      | 8.6min             |

# PII Redaction App
- [Setting Up a Development Environment](docs/development.md)
