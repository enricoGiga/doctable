import copy
import numpy as np
import time

from doctable.table_recognition import predict_det, predict_rec, utility
from doctable.table_recognition.matcher import TableMatch
from doctable.table_recognition.predict_structure import TableStructure
from doctable.table_recognition.predict_system import sorted_boxes


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


class TableSystem(object):
    def __init__(self, args, text_detector=None, text_recognizer=None):
        self.args = args


        self.text_detector = predict_det.TextDetector(copy.deepcopy(
            args)) if text_detector is None else text_detector
        self.text_recognizer = predict_rec.TextRecognizer(copy.deepcopy(
            args)) if text_recognizer is None else text_recognizer

        self.table_structure = TableStructure(args)
        self.match = TableMatch(filter_ocr_result=True)

        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'table')

    def __call__(self, img, return_ocr_result_in_table=False):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()
        structure_res, elapse = self._structure(copy.deepcopy(img))
        result['cell_bbox'] = structure_res[1].tolist()
        time_dict['table'] = elapse

        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
            copy.deepcopy(img))
        time_dict['det'] = det_elapse
        time_dict['rec'] = rec_elapse

        if return_ocr_result_in_table:
            result['boxes'] = [x.tolist() for x in dt_boxes]
            result['rec_res'] = rec_res

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict['match'] = toc - tic
        result['html'] = pred_html
        end = time.time()
        time_dict['all'] = end - start
        return result, time_dict

    def _structure(self, img):
        structure_res, elapse = self.table_structure(copy.deepcopy(img))
        return structure_res, elapse

    def _ocr(self, img):
        h, w = img.shape[:2]
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)

        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)

        return dt_boxes, rec_res, det_elapse, rec_elapse


