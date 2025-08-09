import sys
import os
sys.path.append(os.path.abspath('..'))
from utlis.scan_engine_utlis.scan_engine_utlis import is_special_date

STATUS_FIELDS_CONFIG = {
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

        "mini_6cam_map": { # note this thing is just to find out if there is mapping or not 
        "default": 0,  # NO
        "conditions": [
            {
                # YES if a sync_to_mini_path.txt lives directly in the subfolder
                "condition": lambda **kwargs: os.path.isfile(
                    os.path.join(kwargs["subfolder_path"], "sync_to_mini_path.txt")
                ),
                "value": 1
            },
            {
                # (Optional) mark FAILED if this subfolder was in failed_paths
                "condition": lambda **kwargs: kwargs["subfolder_path"] in kwargs.get("failed_paths", []),
                "value": 3
            }
        ]
    },
        'social': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'social' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "social" is in the path
            },
            {
                'condition': lambda **kwargs: 'mini_p' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "social" is in the path
            },
            {
                'condition': lambda **kwargs: '2mice' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "social" is in the path
            },
                        {
                'condition': lambda **kwargs: '_p' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "social" is in the path
            },
                        {
                'condition': lambda **kwargs: '2male_mice' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "social" is in the path
            },   
        ]
    },
    'miniscope': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'mini' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "mini" is in the path
            },

            {
                'condition': lambda **kwargs: any(
                    term in kwargs['subfolder_path'].lower() for term in ['pmc', 'v1'] #somtimes this is wrong because sometims i am just running the animal for testing.
                ),
                'value': 1
            }
            
        ]
    },
        'pmc': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'mini' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "mini" is in the path
            },

            {
                'condition': lambda **kwargs: any(
                    term in kwargs['subfolder_path'].lower() for term in ['pmc'] #somtimes this is wrong because sometims i am just running the animal for testing.
                ),
                'value': 1
            }
            
        ]
    },    
    'v1': {
        'default': 0,
        'conditions': [
            # {
            #     'condition': lambda **kwargs: 'mini' in kwargs['subfolder_path'].lower(),
            #     'value': 1  # Set to 1 if "mini" is in the path
            # },

            {
                'condition': lambda **kwargs: any(
                    term in kwargs['subfolder_path'].lower() for term in ['v1'] #somtimes this is wrong because sometims i am just running the animal for testing.
                ),
                'value': 1
            }
            
        ]
    },
    'test': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'test' in kwargs['subfolder_path'].lower(),
                'value': 1  # Set to 1 if "test" is in the path
            }
        ]
    },
    'after_oxytocin': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'AO' in kwargs['subfolder_path'].upper(),
                'value': 1  # Set to 1 if "AO" is in the path
            }
        ]
    },
    'before_oxytocin': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: 'BO' in kwargs['subfolder_path'].upper(),
                'value': 1  # Set to 1 if "BO" is in the path
            }
        ]
    },

    'mini_rec_sync':{
        'default': 0,  # Default value when neither file is found
        'conditions': [
            {
                'condition': lambda **kwargs: os.path.exists(
                    os.path.join(kwargs['subfolder_path'], 'MIR_Aligned')) and any(f.startswith('aligned_predictions_with_ca_and_dF_F') and f.endswith('.h5') for f in os.listdir(os.path.join(kwargs['subfolder_path'], 'MIR_Aligned'))
                ),
                'value': 1  # Set to 1 if the _and_dF_F file exists
            },
            {
                'condition': lambda **kwargs: os.path.exists(
                    os.path.join(kwargs['subfolder_path'], 'MIR_Aligned', 'aligned_predictions_with_ca.h5')
                ),
                'value': 0.5  # Set to 0.5 if the basic file exists
            }
        ]
    },

}