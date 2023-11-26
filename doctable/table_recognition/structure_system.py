from doctable.table_recognition.predict_system import TextSystem
from doctable.table_recognition.predict_table import TableSystem


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode

        self.text_system = TextSystem(args)
        self.table_system = TableSystem(
            args, self.text_system.text_detector,
            self.text_system.text_recognizer)

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        time_dict = {
            'image_orientation': 0,
            'layout': 0,
            'table': 0,
            'table_match': 0,
            'det': 0,
            'rec': 0,
            'kie': 0,
            'all': 0
        }

        ori_im = img.copy()

        h, w = ori_im.shape[:2]
        layout_res = [dict(bbox=None, label='table')]
        res_list = []
        for region in layout_res:
            res = ''
            if region['bbox'] is not None:
                x1, y1, x2, y2 = region['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi_img = ori_im[y1:y2, x1:x2, :]
            else:
                x1, y1, x2, y2 = 0, 0, w, h
                roi_img = ori_im
            if region['label'] == 'table':
                if self.table_system is not None:
                    res, table_time_dict = self.table_system(
                        roi_img, return_ocr_result_in_table)
                    time_dict['table'] += table_time_dict['table']
                    time_dict['table_match'] += table_time_dict['match']
                    time_dict['det'] += table_time_dict['det']
                    time_dict['rec'] += table_time_dict['rec']

            res_list.append({
                'type': region['label'].lower(),
                'bbox': [x1, y1, x2, y2],
                'img': roi_img,
                'res': res,
                'img_idx': img_idx
            })
        return res_list
