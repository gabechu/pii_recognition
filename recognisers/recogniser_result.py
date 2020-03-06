from dataclasses import dataclass


@dataclass
class RecogniserResult:
    entity_type: str
    start: int
    end: int
