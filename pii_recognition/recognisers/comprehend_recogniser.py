from typing import List

from boto3.session import Session
from botocore.client import BaseClient
from decouple import config

from pii_recognition.aws.config_session import config_cognito_session
from pii_recognition.labels.schema import Entity

from .entity_recogniser import EntityRecogniser

# read from .env
IDENTITY_POOL_ID = config("IDENTITY_POOL_ID")
AWS_REGION = "us-west-2"


class ComprehendRecogniser(EntityRecogniser):
    def __init__(self, supported_entities: List[str],
                 supported_languages: List[str]):
        sess = config_cognito_session(IDENTITY_POOL_ID, AWS_REGION)
        self.comprehend = self._initiate_comprehend(sess)

        super().__init__(supported_entities=supported_entities,
                         supported_languages=supported_languages)

    def _initiate_comprehend(self, session: Session) -> BaseClient:
        return session.client(service_name="comprehend",
                              region_name=AWS_REGION)

    def analyse(self, text: str, entities: List[str]) -> List[Entity]:
        self.validate_entities(entities)

        # TODO: Add feature supporting multilingual but the first round is for
        # only English
        DEFAULT_LANG = 'en'

        response = self.comprehend.detect_entities(Text=text,
                                                   LanguageCode=DEFAULT_LANG)
        predicted_entities = response["Entities"]

        # Enhancement: filter on comprehend prediction scores
        filtered = filter(lambda ent: ent["Type"] in entities,
                          predicted_entities)
        span_labels = map(
            lambda ent: Entity(ent["Type"], ent["BeginOffset"], ent[
                "EndOffset"]), filtered)
        return list(span_labels)
