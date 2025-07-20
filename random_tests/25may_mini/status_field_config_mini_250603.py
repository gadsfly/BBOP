import sys
import os
sys.path.append(os.path.abspath('..'))
# from utlis.scan_engine_utlis.scan_engine_utlis import is_special_date

STATUS_FIELDS_CONFIG = {

    # 'minian': {
    #     'default': 0,
    #     'conditions': [
    #         {
    #             'condition': lambda **kwargs: os.path.exists(
    #                 os.path.join(kwargs['subfolder_path'], 'minian_dataset.nc')
    #             ),
    #             'value': 1  # Mark 1 if "minian_dataset.nc" is found
    #         }
    #     ]
    # },

    'minian': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: any(
                    fname.endswith('.nc') for fname in os.listdir(kwargs['subfolder_path'])
                ),
                'value': 1  # Mark 1 if any .nc file is found
            }
        ]
        },

    "mini_6cam_map": {  # note this thing is just to find out if there is mapping or not 
        "default": 0,  # NO
        "conditions": [
            {
                # YES if sync_to_rec_path.txt lives in the parent of subfolder_path
                "condition": lambda **kwargs: os.path.isfile(
                    os.path.join(
                        os.path.dirname(kwargs["subfolder_path"]),
                        "sync_to_rec_path.txt"
                    )
                ),
                "value": 1
            }
        ]
    },


"social": {
    "default": 0,
    "conditions": [
        {
            "condition": lambda **kwargs: (
                os.path.isfile(os.path.join(
                    os.path.dirname(kwargs["subfolder_path"]),
                    "sync_to_rec_path.txt"
                )) and
                "social" in open(
                    os.path.join(
                        os.path.dirname(kwargs["subfolder_path"]),
                        "sync_to_rec_path.txt"
                    ), "r"
                ).read().lower()
            ),
            "value": 1
        },
        {
            "condition": lambda **kwargs: (
                os.path.isfile(os.path.join(
                    os.path.dirname(kwargs["subfolder_path"]),
                    "sync_to_rec_path.txt"
                )) and
                "mini_p" in open(
                    os.path.join(
                        os.path.dirname(kwargs["subfolder_path"]),
                        "sync_to_rec_path.txt"
                    ), "r"
                ).read().lower()
            ),
            "value": 1
        },
        {
            "condition": lambda **kwargs: (
                os.path.isfile(os.path.join(
                    os.path.dirname(kwargs["subfolder_path"]),
                    "sync_to_rec_path.txt"
                )) and
                "2mice" in open(
                    os.path.join(
                        os.path.dirname(kwargs["subfolder_path"]),
                        "sync_to_rec_path.txt"
                    ), "r"
                ).read().lower()
            ),
            "value": 1
        },
        {
            "condition": lambda **kwargs: (
                os.path.isfile(os.path.join(
                    os.path.dirname(kwargs["subfolder_path"]),
                    "sync_to_rec_path.txt"
                )) and
                "_p" in open(
                    os.path.join(
                        os.path.dirname(kwargs["subfolder_path"]),
                        "sync_to_rec_path.txt"
                    ), "r"
                ).read().lower()
            ),
            "value": 1
        },
        {
            "condition": lambda **kwargs: (
                os.path.isfile(os.path.join(
                    os.path.dirname(kwargs["subfolder_path"]),
                    "sync_to_rec_path.txt"
                )) and
                "2male_mice" in open(
                    os.path.join(
                        os.path.dirname(kwargs["subfolder_path"]),
                        "sync_to_rec_path.txt"
                    ), "r"
                ).read().lower()
            ),
            "value": 1
        }
    ]
},



    'minian_selected': {
        'default': 0,
        'conditions': [
            {
                'condition': lambda **kwargs: sum(
                    fname.endswith('.nc') for fname in os.listdir(kwargs['subfolder_path'])
                ) == 1,
                'value': 1  # Mark 1 if only one .nc file is found
            }
        ]
    },

    'valid_rec': {
        # If we want “0” for missing 4.avi and “1” if found:
        'default': 0,  
        'conditions': [
            {
                'condition': lambda **kwargs: os.path.exists(
                    os.path.join(kwargs['subfolder_path'], '4.avi')
                ),
                'value': 1
            }
        ]},
    #below are failed tries.    
        'mapped': {
    'default': 0,
    'conditions': [
        {
            'condition': lambda **kwargs: kwargs.get('experiment_path') in kwargs.get('manual_log', {}),
            'value': 1  # Mark as mapped if the experiment path is found in the CSV
        }
    ]
},
'quality': {
    'default': 'unknown',
    'conditions': [
        {
            'condition': lambda **kwargs: kwargs.get('experiment_path') in kwargs.get('manual_log', {}),
            'value': lambda **kwargs: kwargs.get('manual_log')[kwargs.get('experiment_path')]
        }
    ]
},


}