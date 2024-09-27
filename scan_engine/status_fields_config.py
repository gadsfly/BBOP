import sys
import os
sys.path.append(os.path.abspath('..'))
from utlis.scan_engine_utlis.scan_engine_utlis import is_special_date

STATUS_FIELDS_CONFIG = {
    'mir_generate_param': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda calib_file: calib_file is not None,
                'value': 1  # YES
            }
        ]
    },
    'sync': {
        'default': 0,  # Default NO
        'conditions': [
            {
                'condition': lambda calib_file: calib_file and calib_file.startswith("df_") and calib_file.endswith("label3d_dannce.mat"),
                'value': 1  # YES
            },
            {
                'condition': lambda subfolder_path, failed_paths: subfolder_path in failed_paths,
                'value': 3  # FAILED
            }
        ]
    },
    'z_adjusted': {
        'default': 2,  # Default NO-NEED
        'conditions': [
            {
                'condition': lambda folder_name: is_special_date(folder_name),
                'value': 0  # NO
            },
            {
                'condition': lambda subfolder_path: any(file_name.endswith("label3d_dannce.mat.old") for file_name in os.listdir(subfolder_path)),
                'value': 1  # YES
            }
        ]
    }
}
