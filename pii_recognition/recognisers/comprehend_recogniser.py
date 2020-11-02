from typing import Callable, List

from boto3.session import Session
from botocore.client import BaseClient
from decouple import config
from pii_recognition.aws.config_session import config_cognito_session
from pii_recognition.labels.schema import Entity

from .entity_recogniser import EntityRecogniser


class ModelMapping(dict):
    def __getitem__(self, key: str) -> Callable:
        try:
            return super().__getitem__(key)
        except KeyError:
            key_list = list(super().keys())
            raise ValueError(
                f"Available model names are: {key_list} but got model named {key}"
            )


class ComprehendRecogniser(EntityRecogniser):
    # read from .env
    IDENTITY_POOL_ID = config("IDENTITY_POOL_ID")
    AWS_REGION = "us-west-2"

    def __init__(
        self,
        supported_entities: List[str],
        supported_languages: List[str],
        model_name: str,
    ):
        sess = config_cognito_session(self.IDENTITY_POOL_ID, self.AWS_REGION)
        comprehend = self._initiate_comprehend(sess)
        model_mapping = ModelMapping(
            ner=comprehend.detect_entities, pii=comprehend.detect_pii_entities
        )
        self.model_func = model_mapping[model_name]
        self.model_name = model_name

        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    def _initiate_comprehend(self, session: Session) -> BaseClient:
        return session.client(service_name="comprehend", region_name=self.AWS_REGION)

    def analyse(self, text: str, entities: List[str]) -> List[Entity]:
        self.validate_entities(entities)

        # TODO: Add multilingual support
        # based on boto3 Comprehend doc Comprehend supports
        # 'en'|'es'|'fr'|'de'|'it'|'pt'|'ar'|'hi'|'ja'|'ko'|'zh'|'zh-TW'
        DEFAULT_LANG = "en"

        response = self.model_func(Text=text, LanguageCode=DEFAULT_LANG)

        # parse response
        predicted_entities = response["Entities"]
        # Remove entities we are not interested
        filtered = filter(lambda ent: ent["Type"] in entities, predicted_entities)
        span_labels = map(
            lambda ent: Entity(ent["Type"], ent["BeginOffset"], ent["EndOffset"]),
            filtered,
        )

        return list(span_labels)
