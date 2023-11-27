from typing import Union

import numpy as np

from doctable.table_recognition.PPStructure import PPStructure
from doctable.data_utility.utilities import cv2_read


class TableRecognizer:

    def __init__(self):
        self.ppstructure = PPStructure()

    def recognize(self, cropped_image: Union[str, np.array]):
        if isinstance(cropped_image, str):
            cropped_image = cv2_read(cropped_image)

        return self.ppstructure.__call__(img=cropped_image,
                                         return_ocr_result_in_table=True)
