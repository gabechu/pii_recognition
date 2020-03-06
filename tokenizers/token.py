from dataclasses import dataclass


@dataclass
class Token:
    data: str
    start: int
    end: int
