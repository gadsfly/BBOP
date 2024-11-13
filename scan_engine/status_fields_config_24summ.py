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

    'saline': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'saline' in kwargs['subfolder_path'].lower() or 'pb' in kwargs['subfolder_path'].lower(),
                'value': 1
            }
        ]
    },

    'caffeine': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'caffeine' in kwargs['subfolder_path'].lower() or 'cf' in kwargs['subfolder_path'].lower(),
                'value': 1
            }
        ]
    },
    'first': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: any(keyword in kwargs['subfolder_path'].upper() for keyword in ['PBF', 'PBS', 'CFF', 'CFS']),
                'value': 1
            }
        ]
    },
    'second': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: any(keyword in kwargs['subfolder_path'].upper() for keyword in ['PBF', 'PBS', 'CFF', 'CFS']),
                'value': 1
            }
        ]
    },
    'habituation': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'hbn' in kwargs['subfolder_path'].upper() or 'habituation' in kwargs['subfolder_path'].lower(),
                'value': 1
            }
        ]
    },
    # 'dropf_handle': {
    #     'default': 0,  # Default NO
    #     'conditions': [
    #         {
    #             'condition': lambda **kwargs: kwargs.get('calib_file') and os.path.basename(kwargs['calib_file']).startswith("df_dh_") and kwargs['calib_file'].endswith("label3d_dannce.mat"),
    #             'value': 1  # YES
    #         },
    #         {
    #             'condition': lambda **kwargs: kwargs.get('subfolder_path') in kwargs.get('failed_paths', []),
    #             'value': 3  # FAILED
    #         },
    #         {
    #             'condition': lambda **kwargs: kwargs.get('date_folder') and kwargs['date_folder'] <= '2024_11_01' and not (
    #                 kwargs.get('calib_file') and os.path.basename(kwargs['calib_file']).startswith("df_dh_") and kwargs['calib_file'].endswith("label3d_dannce.mat")
    #             ),
    #             'value': 2  # NO NEED
    #         }
    #     ]
    # },


    'com': {
        'default': 0,  # Default NO
        'conditions': [
            {
                'condition': lambda **kwargs: os.path.exists(os.path.join(kwargs['subfolder_path'], 'COM/predict00')) and
                                               any(f.startswith('com3d') and f.endswith('.mat')
                                                   for f in os.listdir(os.path.join(kwargs['subfolder_path'], 'COM/predict00'))),
                'value': 1  # YES
            },
            {
                'condition': lambda **kwargs: kwargs['subfolder_path'] in kwargs['failed_paths'],
                'value': 3  # FAILED
            }
        ]
    },
    
    'dannce': {
        'default': 0,  # Default NO
        'conditions': [
            {
                'condition': lambda **kwargs: os.path.exists(os.path.join(kwargs['subfolder_path'], 'DANNCE/predict00')) and
                                            'save_data_AVG.mat' in os.listdir(os.path.join(kwargs['subfolder_path'], 'COM/predict00')),
                'value': 1  # YES
            }
        ]
    },
    # 'social': {
    #     'default': 0,
    #     'conditions': [
    #         {
    #             'condition': lambda **kwargs: 'social' in kwargs['subfolder_path'].lower(),
    #             'value': 1  # Set to 1 if "social" is in the path
    #         },
    #         {
    #             'condition': lambda **kwargs: '2mice' in kwargs['subfolder_path'].lower(),
    #             'value': 1  # Set to 1 if "social" is in the path
    #         }            
    #     ]
    # },
    # 'miniscope': {
    #     'default': 0,
    #     'conditions': [
    #         {
    #             'condition': lambda **kwargs: 'mini' in kwargs['subfolder_path'].lower(),
    #             'value': 1  # Set to 1 if "mini" is in the path
    #         }
    #     ]
    # },
    'test': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'test' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "test" is in the path
            }
        ]
    },
    # 'after_oxytocin': {
    #     'default': 0,
    #     'conditions': [
    #         {
    #             'condition': lambda **kwargs: 'ao' in kwargs['subfolder_path'].upper(),
    #             'value': 1  # Set to 1 if "AO" is in the path
    #         }
    #     ]
    # },
    # 'before_oxytocin': {
    #     'default': 0,
    #     'conditions': [
    #         {
    #             'condition': lambda **kwargs: 'bo' in kwargs['subfolder_path'].upper(),
    #             'value': 1  # Set to 1 if "BO" is in the path
    #         }
    #     ]
    # }

    
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

