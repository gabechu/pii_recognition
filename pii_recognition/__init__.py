__version__ = "0.1.0"

from pii_recognition.registration.bind import (  # noqa: F401
    _register_recognisers,
    registry,
)

_register_recognisers()
