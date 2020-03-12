from dataclasses import dataclass


@dataclass
class Token:
    text: str
    start: int
    end: int
