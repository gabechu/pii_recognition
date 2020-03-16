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
        failed: Whether failed to extract sample errors of the given text. Length
            mismatch between predicted tokens and annotated tokens can cause the
            failure.
    """

    token_errors: List[TokenError]
    full_text: str
    failed: bool
