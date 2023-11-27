import os
from typing import List

import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sahi import ObjectPrediction
from sahi.utils.cv import read_image
from ultralytics.yolo.engine.results import Results
from ultralyticsplus import YOLO

from doctable.data_utility.datatype import Table
from doctable.table_detection.detect_utils import get_object_predictions, \
    is_pil_image, is_str_image, get_pil_image


class TableDetector:
    """
    This class is used for table detection
    """

    def __init__(self):
        """
        This method will initialize the table detector model and set the model parameters
        """
        self.model = YOLO('foduucom/table-detection-and-extraction')

        # set model parameters
        self.model.overrides['conf'] = 0.25  # NMS confidence threshold
        self.model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 1000  # maximum number of detections per image
        self.object_predictions: List[ObjectPrediction] = []

    def predict(self, image) -> List[Results]:
        """
        This method will return the prediction results for the given image
        :param image: the image for table detection
        :return: a list of prediction results for the given image
        """

        return self.model.predict(image)

    def get_cropped_tables(self, image, x_padding=30,
                           y_padding=10) -> List[Table]:
        """
        This method will return a list of cropped tables from the given image
        :param image: the image for table detection
        :param x_padding: the padding for x-axis
        :param y_padding: the padding for y-axis
        :return: a list of cropped tables from the given image
        """
        predictions = self.predict(image)
        self.set_object_predictions(image, predictions)
        tables = []
        for object_prediction in self.object_predictions:
            x_1 = int(object_prediction.bbox.minx - x_padding)
            y_1 = int(object_prediction.bbox.miny - y_padding)
            pt1 = (x_1, y_1)
            x_2 = int(object_prediction.bbox.maxx + x_padding)
            y_2 = int(object_prediction.bbox.maxy + y_padding)
            pt2 = (x_2, y_2)
            if is_pil_image(image):
                nparray = np.array(image)
            elif is_str_image(image):
                nparray = read_image(image)
            else:
                raise ValueError(
                    "Input image should be a PIL Image or a string representing the image file path.")
            image_cropped = nparray[y_1:y_2, x_1:x_2]

            tables.append(Table(pt1, pt2, image_cropped))
        return tables

    def set_object_predictions(self, image, predictions):
        """
        This method will set the object predictions for the given image
        :param image: the image for table detection
        :param predictions: the predictions for the given image
        :return: None
        """
        self.object_predictions = get_object_predictions(model=self.model, image=image,
                                                         result=predictions[0])

    def show_detection_results(self, img_path: str, x_padding=5, y_padding=5) -> None:
        """
        This method will show the detection results on the given image
        :param img_path: the image for table detection
        :param x_padding: the padding for x-axis
        :param y_padding: the padding for y-axis
        :return: None
        """

        if os.path.basename(img_path)[-3:].lower() == 'pdf':
            doc = fitz.open(img_path)
            for page in doc:  # iterate over document pages
                pix = page.get_pixmap(dpi=150)  # render full page with desired DPI
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                d_tables = self.get_cropped_tables(image=img, x_padding=x_padding,
                                                   y_padding=y_padding)
                for d_table in d_tables:
                    plt.imshow(d_table.cropped_image)
                    plt.show()
                    img = cv2.rectangle(img, d_table.pt1, d_table.pt2, (255, 0, 0),
                                        thickness=2)
                plt.imshow(img)
                # plt.savefig(f"{os.environ['PROJECT_DIR']}/data/images/mygraph.png")
        else:

            img = get_pil_image(img_path)
            d_tables = self.get_cropped_tables(image=img, x_padding=x_padding,
                                               y_padding=y_padding)
            for d_table in d_tables:
                plt.imshow(d_table.cropped_image)

                img = cv2.rectangle(np.array(img), d_table.pt1, d_table.pt2,
                                    (255, 0, 0), thickness=2)
            plt.imshow(img)
            plt.show()
            # plt.savefig(f"{os.environ['PROJECT_DIR']}/data/images/mygraph.png")


if __name__ == '__main__':
    pdf_path = f"{os.environ['PROJECT_DIR']}/data/images/detection_img1.jpg"
    table_detector = TableDetector()
    table_detector.show_detection_results(pdf_path)
