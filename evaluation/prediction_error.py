from dataclasses import dataclass
from typing import List


@dataclass
class TokenError:
    annotation: str
    prediction: str
    token: str


@dataclass
class SampleError:
    token_errors: List[TokenError]
    full_text: str
    length_error: bool
