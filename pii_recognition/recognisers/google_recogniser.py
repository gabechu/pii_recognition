from typing import Dict, List

from decouple import config
from google.cloud import language_v1
from google.cloud.language_v1 import LanguageServiceClient
from pii_recognition.labels.schema import Entity

from .entity_recogniser import EntityRecogniser


class GoogleRecogniser(EntityRecogniser):
    CREDENTIALS_PATH = config("GOOGLE_APPLICATION_CREDENTIALS")

    def __init__(self, supported_entities: List[str], supported_languages: List[str]):
        super().__init__(
            supported_entities=supported_entities,
            supported_languages=supported_languages,
        )

    @property
    def request_template(self) -> Dict:
        # supported languages: zh zh-Hant en fr de it ja ko pt ru es
        # https://cloud.google.com/natural-language/docs/languages
        return {
            "document": {
                "content": None,  # it's a placeholder
                "type_": language_v1.Document.Type.PLAIN_TEXT,
                "language": "en",  # Start with english
            },
            "encoding_type": language_v1.EncodingType.UTF8,
        }

    @property
    def client(self) -> LanguageServiceClient:
        return language_v1.LanguageServiceClient.from_service_account_json(
            self.CREDENTIALS_PATH
        )

    def analyse(self, text: str, entities: List[str]) -> List[Entity]:
        self.validate_entities(entities)

        # this avoids mutating state of request template
        request = self.request_template
        request["document"]["content"] = text

        response = self.client.analyze_entities(request)

        # parse response
        span_labels = []
        for entity in response.entities:
            entity_type = entity.type_.name
            if entity_type in entities:
                for mention in entity.mentions:
                    start = mention.text.begin_offset
                    text_length = len(mention.text.content)
                    end = start + text_length
                    span_labels.append(
                        Entity(entity_type=entity_type, start=start, end=end)
                    )

        return span_labels
