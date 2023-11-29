from typing import Union

import numpy as np

from src.table_recognition.PPStructure import PPStructure
from src.data_utility.utilities import cv2_read


class TableRecognizer:
    """
    The TableRecognizer class is responsible for recognizing tables in a given image.
    It uses the PPStructure class to perform the recognition.
    """

    def __init__(self):
        """
        Initializes the TableRecognizer class with a PPStructure.
        """
        self.ppstructure = PPStructure()

    def recognize(self, cropped_image: Union[str, np.array]):
        """
        Recognizes tables in a given image. The image can be a path to an image file or a numpy array.

        Args:
            cropped_image (Union[str, np.array]): The image to recognize tables in.

        Returns:
            The result of the table recognition.
        """
        if isinstance(cropped_image, str):
            cropped_image = cv2_read(cropped_image)

        return self.ppstructure.__call__(img=cropped_image,
                                         return_ocr_result_in_table=True)
