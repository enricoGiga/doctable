import os
from typing import List, Union

import fitz
from PIL import Image, UnidentifiedImageError

from doctable.data_utility.datatype import Page
from doctable.table_detection.detection import TableDetector
from doctable.table_recognition.recognition import TableRecognizer


class Doctable:
    def __init__(self):
        self.detector = TableDetector()
        self.recognizer = TableRecognizer()

    def process_page(self, page: Union[fitz.Page, Image.Image], page_number: int) -> Page:
        new_page = Page(page_number=page_number)

        if isinstance(page, fitz.Page):
            pix = page.get_pixmap(dpi=150)
            page = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        tables = self.detector.get_cropped_tables(image=page)
        for j, table in enumerate(tables):
            res = self.recognizer.recognize(table.cropped_image)
            for result in res:
                table.recognition_results = result["res"]
                new_page.tables.append(table)
        return new_page

    def table_extraction(self, path: str) -> List[Page]:
        pages: List[Page] = []

        try:
            if os.path.basename(path)[-3:].lower() == 'pdf':
                doc = fitz.open(path)
                for i, page in enumerate(doc):
                    pages.append(self.process_page(page, i + 1))
            else:
                img = Image.open(path)
                pages.append(self.process_page(img, 1))
        except (FileNotFoundError, IsADirectoryError, UnidentifiedImageError):
            print(
                f"Error: Could not open file {path}. Please make sure the file exists "
                f"and is a valid PDF file or image.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return pages
