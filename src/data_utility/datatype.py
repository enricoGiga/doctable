from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np


@dataclass
class Table:
    """
    The Table class represents a table detected in a document.

    Attributes:
        pt2 (Tuple[int, int]): The top right point of the bounding box around the table.
        pt1 (Tuple[int, int]): The bottom left point of the bounding box around the table.
        cropped_image (np.array): The cropped image of the table.
        recognition_results (List): The results of the table recognition.
    """
    pt2: Tuple[int, int]
    pt1: Tuple[int, int]
    cropped_image: np.array
    recognition_results = []


@dataclass
class Page:
    """
    The Page class represents a page in a document.

    Attributes:
        page_number (int): The number of the page in the document.
        tables (List[Table]): A list of tables detected in the page.
    """
    page_number: int
    tables: List[Table] = field(default_factory=list)
