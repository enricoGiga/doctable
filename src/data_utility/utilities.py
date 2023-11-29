import inspect
from pathlib import Path

import cv2
import numpy as np


def cv2_read(img_path: str) -> np.ndarray:
    """
    Reads an image using cv2.
    :param img_path: the path to the image
    :return: np.ndarray: the image
    """
    full_img = cv2.imread(img_path)
    # Convert the image from BGR (cv2 default loading style) to RGB
    full_img = np.array(full_img[..., ::-1])
    return full_img


def get_caller_directory_path() -> Path:
    """
    This function returns the absolute path of the directory of the script that calls this function.
    It uses the inspect module to get the call stack and the pathlib module to construct and manipulate the path.
    :return: pathlib.Path: The absolute path of the directory of the script that calls this function.
    """
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    caller_path = Path(caller_module.__file__).resolve()
    return caller_path


def get_parent_directory_path(levels_up: int) -> Path:
    """
    This function returns the absolute path of a directory that is 'level_up' levels above
     the current directory.
    It uses the pathlib module to construct and manipulate the path.

    :param levels_up: int: The number of levels up from the current directory to go.
    :return: pathlib.Path: The absolute path of the directory 'level_up' levels above the
    current directory.
    """
    caller_dir_path = get_caller_directory_path()
    return caller_dir_path.parents[levels_up]


def get_project_directory_path() -> Path:
    """
    Get project root directory path.
    It uses the pathlib module to construct and manipulate the path.

    :return: pathlib.Path: The absolute path of project root directory.
    """
    abs_path_current_dir = Path(__file__).resolve()
    return abs_path_current_dir.parents[2]
