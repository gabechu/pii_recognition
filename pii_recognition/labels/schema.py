from __future__ import annotations  # class forward reference

from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class Entity:
    entity_type: str
    start: int
    end: int


@dataclass
class TokenLabel:
    entity_type: str
    start: int
    end: int


# TODO: consider to change to dataclass for consistency
class EvalLabel(NamedTuple):
    annotated: str
    predicted: str
