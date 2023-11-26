import os
import sys

path_dict = {os.path.basename(path): path for path in sys.path}


def get_path_by_name(name):
    return path_dict.get(name, None)


os.environ["PROJECT_ROOT"] = get_path_by_name("doctable")

