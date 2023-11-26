from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np


@dataclass
class Table:
    pt2: Tuple[int, int]
    pt1: Tuple[int, int]
    cropped_image: np.array
    recognition_results = []


@dataclass
class Page:
    page_number: int
    tables: List[Table] = field(default_factory=list)
