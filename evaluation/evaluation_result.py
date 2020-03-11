from dataclasses import dataclass
from collections import Counter
from .prediction_error import SampleError


@dataclass
class EvaluationResult:
    label_pair: Counter  # annotation and prediction in tuple
    mistakes: SampleError
