from pii_recognition.registration.registry import Registry

from .conll_reader import ConllReader
from .reader import Reader
from .wnut_reader import WnutReader


def init() -> Registry:
    registry = Registry[Reader]()
    registry.register(ConllReader)
    registry.register(WnutReader)

    return registry


reader_registry: Registry[Reader] = init()
