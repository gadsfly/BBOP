import os
import re

# this is written to remove everything besides a folder called videos and any other folders containing "label"
# calib will be removed. 

#this will loop through everything, no need to get date as a key

base_path = '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ'

def clean_up(base_path):
    pattern = re.compile(r'^\d')
    # for anything in format of XXXX_XXXX_XX, where XX are int:
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        # Check if the item is a directory and starts with an integer
        if os.path.isdir(item_path) and pattern.match(item):
    # the loop through each folder that starts with int. do not loop through folders start with letter.