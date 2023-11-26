from doctable.table_recognition.init_args import init_args
from doctable.table_recognition.structure_system import StructureSystem
from doctable.table_recognition.utility import check_img


class PPStructure(StructureSystem):
    def __init__(self, **kwargs):
        params, unknown = init_args().parse_known_args()
        params.__dict__.update(**kwargs)
        params.mode = 'structure'
        super().__init__(params)

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        img = check_img(img)
        res = super().__call__(
            img, return_ocr_result_in_table, img_idx=img_idx)
        return res
