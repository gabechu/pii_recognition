from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class SpanLabel:
    entity_type: str
    start: int
    end: int


class EvalLabel(NamedTuple):
    annotated: str
    predicted: str
