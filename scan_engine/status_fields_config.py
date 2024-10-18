import sys
import os
sys.path.append(os.path.abspath('..'))
from utlis.scan_engine_utlis.scan_engine_utlis import is_special_date

STATUS_FIELDS_CONFIG = {
    'mir_generate_param': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: kwargs['calib_file'] is not None,
                'value': 1  # YES
            }
        ]
    },
    'sync': {
        'default': 0,  # Default NO
        'conditions': [
            {
                'condition': lambda **kwargs: kwargs['calib_file'] and os.path.basename(kwargs['calib_file']).startswith("df_") and kwargs['calib_file'].endswith("label3d_dannce.mat"),
                'value': 1  # YES
            },
            {
                'condition': lambda **kwargs: kwargs['subfolder_path'] in kwargs['failed_paths'],
                'value': 3  # FAILED
            }
        ]
    },
    # 'z_adjusted': {
    #     'default': 2,  # Default NO-NEED
    #     'conditions': [
    #         {
    #             'condition': lambda **kwargs: is_special_date(kwargs['folder_name']),
    #             'value': 0  # NO
    #         },
    #         {
    #             'condition': lambda **kwargs: any(file_name.endswith("label3d_dannce.mat.old") for file_name in os.listdir(kwargs['subfolder_path'])),
    #             'value': 1  # YES
    #         }
    #     ]
    # }
}

