from pii_recognition.registration.registry import Registry
from pii_recognition.tokenisation.detokenisers import (
    Detokeniser,
    SpaceJoinDetokeniser,
    TreebankWordDetokeniser,
)
from pii_recognition.tokenisation.tokenisers import Tokeniser, TreebankWordTokeniser


def tokeniser_init() -> Registry:
    registry = Registry[Tokeniser]()
    registry.add_item(TreebankWordTokeniser)

    return registry


def detokeniser_init() -> Registry:
    registry = Registry[Detokeniser]()
    registry.add_item(SpaceJoinDetokeniser)
    registry.add_item(TreebankWordDetokeniser)

    return registry


# initialisations are simple once become complex considering
# separate the two
tokeniser_registry: Registry[Tokeniser] = tokeniser_init()
detokeniser_registry: Registry[Detokeniser] = detokeniser_init()
