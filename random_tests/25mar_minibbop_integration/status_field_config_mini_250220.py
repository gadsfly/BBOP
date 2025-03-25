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