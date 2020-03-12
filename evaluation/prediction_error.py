from dataclasses import dataclass
from typing import List


@dataclass
class TokenError:
    annotation: str
    prediction: str
    token: str


@dataclass
class SampleError:
    """
    Log prediction errors happening in a sample text.

    Attributes:
        token_errors: Prediction error per token.
        full_text: The original sample text being used for prediction.
        length_mismatch: Whether the length of predictions equals to the length of
            annotations. A mismatch occurs due to tokenisation. Different tokenisation
            strategies produce different outcomes.
    """
    token_errors: List[TokenError]
    full_text: str
    length_mismatch: bool
