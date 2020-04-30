from pii_recognition.registration.registry import Registry

from .conll_reader import ConllReader
from .reader import Reader
from .wnut_reader import WnutReader


def init() -> Registry:
    registry = Registry[Reader]()
    registry.add_item(ConllReader)
    registry.add_item(WnutReader)

    return registry


reader_registry: Registry[Reader] = init()
