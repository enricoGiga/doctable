import argparse
import os

from src.data_utility.utilities import get_project_directory_path


def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")


def init_args():
    parser = argparse.ArgumentParser()
    project_root = get_project_directory_path()
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--drop_score", type=float, default=0.5)
    parser.add_argument("--det_box_type", type=str, default='quad')
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    parser.add_argument("--use_space_char", type=str2bool, default=True)

    # params for output
    parser.add_argument("--output", type=str, default='./output')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument(
        "--merge_no_span_structure", type=str2bool, default=True)

    # params for inference
    parser.add_argument(
        "--mode",
        type=str,
        choices=['structure'],
        default='structure',
        help='structure is supported')

    parser.add_argument(
        "--rec_char_dict_path", type=str,
        default=f"{project_root}/configs/en_dict.txt")
    parser.add_argument("--type", type=str, default='structure')
    parser.add_argument("--layout", type=str2bool, default=False)
    parser.add_argument("--det_model_dir", type=str,
                        default=f"{project_root}/data/models/en_PP-OCRv3_det_infer")
    parser.add_argument("--rec_model_dir", type=str,
                        default=f"{project_root}/data/models/en_PP-OCRv3_rec_infer")
    parser.add_argument("--table_model_dir", type=str,
                        default=f"{project_root}/data/models/en_ppstructure_mobile_v2.0_SLANet_infer")
    parser.add_argument("--table_char_dict_path", type=str,
                        default=f"{project_root}/configs/table_structure_dict.txt")

    return parser
