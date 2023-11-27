from typing import List

import numpy as np
from PIL import Image
from sahi import ObjectPrediction
from sahi.prediction import PredictionScore
from sahi.utils.cv import get_bool_mask_from_coco_segmentation, read_image_as_pil
from ultralytics.yolo.engine.results import Results
from ultralyticsplus import YOLO


def is_pil_image(image):
    return isinstance(image, Image.Image)


def is_numpy_image(image):
    return isinstance(image, np.ndarray)


def is_str_image(image):
    return isinstance(image, str)

def get_pil_image(image_path: str) -> Image:
    """
    This function takes an image file path as input and returns a PIL Image object.
    """
    return Image.open(image_path)
def get_object_predictions(
        image,
        model: YOLO,
        result: Results
) -> List[ObjectPrediction]:
    """
    Renders predictions on the image

    Args:
        image (str, URL, Image.Image): image to be rendered
        model (YOLO): YOLO model
        result (ultralytics.yolo.engine.result.Result): output of the model. This is the output of the model.predict() method.

    Returns:
        Image.Image: Image with predictions
    """
    if model.overrides["task"] not in ["detect", "segment"]:
        raise ValueError(
            f"Model task must be either 'detect' or 'segment'. Got {model.overrides['task']}"
        )

    image = read_image_as_pil(image)
    np_image = np.ascontiguousarray(image)

    names = model.model.names

    masks = result.masks
    boxes = result.boxes

    object_predictions = []
    if boxes is not None:
        det_ind = 0
        for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if masks:
                img_height = np_image.shape[0]
                img_width = np_image.shape[1]
                segments = masks.segments
                segments = segments[det_ind]  # segments: np.array([[x1, y1], [x2, y2]])
                # convert segments into full shape
                segments[:, 0] = segments[:, 0] * img_width
                segments[:, 1] = segments[:, 1] * img_height
                segmentation = [segments.ravel().tolist()]

                bool_mask = get_bool_mask_from_coco_segmentation(
                    segmentation, width=img_width, height=img_height
                )
                if sum(sum(bool_mask == 1)) <= 2:
                    continue
                object_prediction = ObjectPrediction.from_coco_segmentation(
                    segmentation=segmentation,
                    category_name=names[int(cls)],
                    category_id=int(cls),
                    full_shape=[img_height, img_width],
                )
                object_prediction.score = PredictionScore(value=conf)
            else:
                object_prediction = ObjectPrediction(
                    bbox=xyxy.tolist(),
                    category_name=names[int(cls)],
                    category_id=int(cls),
                    score=conf,
                )
            object_predictions.append(object_prediction)
            det_ind += 1
        return object_predictions
