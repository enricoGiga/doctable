import os
from typing import List, Union

import fitz
from PIL import Image, UnidentifiedImageError

from src.data_utility.datatype import Page
from src.table_detection.detection import TableDetector
from src.table_recognition.recognition import TableRecognizer


class Doctable:
    """
    The Doctable class is responsible for processing documents and extracting tables from them.
    It uses the TableDetector and TableRecognizer classes to detect and recognize tables respectively.
    """

    def __init__(self):
        """
        Initializes the Doctable class with a TableDetector and TableRecognizer.
        """
        self.detector = TableDetector()
        self.recognizer = TableRecognizer()

    def process_page(self, page: Union[fitz.Page, Image.Image], page_number: int) -> Page:
        """
        Processes a single page from a document. If the page is a fitz.Page, it is converted to an Image.
        Tables are detected and recognized in the page.

        Args:
            page (Union[fitz.Page, Image.Image]): The page to process.
            page_number (int): The number of the page in the document.

        Returns:
            Page: The processed page with recognized tables.
        """
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
        """
        Extracts tables from a document. The document can be a PDF or an image.

        Args:
            path (str): The path to the document.

        Returns:
            List[Page]: A list of processed pages with recognized tables.

        Raises:
            FileNotFoundError, IsADirectoryError, UnidentifiedImageError: If the file does not exist or is not a valid PDF or image.
            Exception: For any other unexpected errors.
        """
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
