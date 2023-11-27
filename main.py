import os
from typing import List

import fitz
import numpy as np
from PIL import Image

from doctable.data_utility.datatype import Page
from doctable.table_detection.detection import TableDetector
from doctable.table_recognition.recognition import TableRecognizer


def table_extraction(path: str) -> List[Page]:
    """

    Args:
        path (str): the path can be either an image or a pdf

    Returns:
        a list of table results for each page (only one page if the :path: is an image instead of a pdf)
    """
    detector = TableDetector()
    recognizer = TableRecognizer()
    pages: List[Page] = []
    if os.path.basename(path)[-3:].lower() == 'pdf':
        doc = fitz.open(path)

        for i, page in enumerate(doc):
            new_page = Page(page_number=i + 1)
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            npImage = np.array(img)

            tables = detector.get_cropped_tables(image=npImage)
            for j, table in enumerate(tables):
                res = recognizer.recognize(table.cropped_image)
                for result in res:
                    table.recognition_results = result["res"]
                    new_page.tables.append(table)
            pages.append(new_page)
    else:
        new_page = Page(page_number=1)
        tables = detector.get_cropped_tables(image=path)
        for j, table in enumerate(tables):
            results = recognizer.recognize(table.cropped_image)
            for result in results:
                table.recognition_results = result["res"]
                new_page.tables.append(table)
        pages.append(new_page)

    return pages


if __name__ == '__main__':
    image_path = f"{os.environ['PROJECT_DIR']}/data/images/detection_img1.jpg"
    total_res = table_extraction(image_path)
    print(total_res)
