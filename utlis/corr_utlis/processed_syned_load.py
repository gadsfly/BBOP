import h5py
import pandas as pd

def load_filtered_data_from_h5(h5_file_path):
    """
    Loads the 'filtered_data' group from an HDF5 file into a Pandas DataFrame.
    """
    with h5py.File(h5_file_path, "r") as hf:
        # Access the group where the DataFrame columns were saved
        grp = hf["filtered_data"]
        
        # Each dataset in 'grp' is a column; read them into a dict of numpy arrays
        data_dict = {}
        for col_name in grp.keys():
            # Read dataset
            col_data = grp[col_name][:]
            
            # If it's a string dataset, h5py might already give you Python strings,
            # but just to be safe, handle bytes if encountered
            if col_data.dtype.kind in ["S", "O"] or isinstance(col_data.flatten()[0], bytes):
                col_data = [elem.decode("utf-8") if isinstance(elem, bytes) else elem 
                            for elem in col_data]
            
            data_dict[col_name] = col_data
        
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data_dict)
    return df


import json
import h5py
import numpy as np
import pandas as pd
import os

def export_aligned_data_to_h5(
    data_obj, 
    rec_path, 
    frame_mapping_file, 
    out_file
):
    """
    Filters data_obj to a specific rec_path and frames from frame_mapping_file,
    then saves minimal data to an HDF5 file (with string columns stored
    as variable-length UTF-8 strings).
    """
    
    # 1. Filter by Prediction_path
    rec_mat_path = os.path.join(rec_path, "DANNCE/predict00/save_data_AVG.mat")
    path_data = data_obj.data[data_obj.data["Prediction_path"] == rec_mat_path].copy()
    if path_data.empty:
        print(f"No data found for rec_path: {rec_mat_path}")
        return
    
    # 2. Read the frame mapping JSON
    with open(frame_mapping_file, "r") as f:
        map_data = json.load(f)
    mapped_frames = set(map_data["mapped_sixcam_frame_indices"])
    time_offset = map_data["time_offset"]
    
    # 3. Filter rows by mapped frame indices (assuming your DataFrame has 'frame' column)
    if "frame" not in path_data.columns:
        raise ValueError("DataFrame does not have 'frame' column to filter by.")
    
    # Adjust frames so that they start at 0
    min_frame = path_data["frame"].min()
    path_data["frame"] = path_data["frame"] - min_frame

    # 1. Create helper offsets DataFrame
    offsets = pd.DataFrame({'offset': range(10)})

    # 2. Cross-merge (pandas 1.2+ supports `how="cross"`)
    expanded = path_data.merge(offsets, how='cross')

    # 3. Update the frame by adding the offset
    expanded['frame'] = expanded['frame'] + expanded['offset']
    expanded.drop(columns='offset', inplace=True)
    
    filtered_data = expanded[expanded["frame"].isin(mapped_frames)]
    
    if filtered_data.empty:
        print("No overlapping frames found between path_data and mapped_sixcam_frame_indices.")
    
    # 4. Save to HDF5
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # Use variable-length string dtype for columns that are (or become) text
    variable_length_string_dt = h5py.special_dtype(vlen=str)
    
    with h5py.File(out_file, "w") as hf:
        
        #
        # (A) Save filtered DataFrame columns
        #
        grp = hf.create_group("filtered_data")
        
        for col in filtered_data.columns:
            col_data = filtered_data[col].to_numpy()  # get as NumPy array
            
            # Check if it's string-like (object, unicode, or bytes)
            if col_data.dtype.kind in ["O", "U", "S"]:
                # Convert each element to a Python string, store as variable-length UTF-8
                # Flatten, map to str, then reshape to original shape if multi-dimensional
                original_shape = col_data.shape
                col_data = col_data.reshape(-1)  # flatten
                col_data = np.array([str(item) for item in col_data], dtype=object)
                col_data = col_data.reshape(original_shape)
                
                grp.create_dataset(
                    col, 
                    data=col_data, 
                    dtype=variable_length_string_dt, 
                    compression="gzip"
                )
            else:
                # Numeric or other supported dtype can be written directly
                grp.create_dataset(col, data=col_data, compression="gzip")
        
        #
        # (B) Save relevant data_obj attributes
        #
        if hasattr(data_obj, "embed_vals") and data_obj.embed_vals is not None:
            hf.create_dataset("embed_vals", data=data_obj.embed_vals, compression="gzip")
        
        # Example: saving meta info
        if hasattr(data_obj, "meta") and data_obj.meta is not None:
            meta_grp = hf.create_group("meta")
            if isinstance(data_obj.meta, dict):
                for key, val in data_obj.meta.items():
                    # Convert to array for consistency
                    val_array = np.array(val, dtype=object)  # object to handle strings
                    # If it has any string/unicode, cast them properly
                    if val_array.dtype.kind in ["O", "U", "S"]:
                        val_array = val_array.reshape(-1)
                        val_array = np.array([str(item) for item in val_array], dtype=object)
                        # Reshape back if needed (only if it's consistent)
                        # But typically meta might be a 1D list, so might not need reshape
                        
                        meta_grp.create_dataset(
                            key,
                            data=val_array,
                            dtype=variable_length_string_dt,
                            compression="gzip"
                        )
                    else:
                        meta_grp.create_dataset(key, data=val_array, compression="gzip")
            else:
                # Non-dict meta structure
                meta_vals = np.array(data_obj.meta, dtype=object)
                if meta_vals.dtype.kind in ["O", "U", "S"]:
                    meta_vals = [str(item) for item in meta_vals.flatten()]
                    meta_vals = np.array(meta_vals, dtype=object)
                    meta_grp.create_dataset(
                        "meta_data",
                        data=meta_vals,
                        dtype=variable_length_string_dt,
                        compression="gzip"
                    )
                else:
                    meta_grp.create_dataset("meta_data", data=meta_vals, compression="gzip")
        
        #
        # (C) Save frame mapping info
        #
        map_grp = hf.create_group("frame_mapping")
        map_grp.create_dataset(
            "mapped_sixcam_frame_indices",
            data=np.array(map_data["mapped_sixcam_frame_indices"]),
            compression="gzip"
        )
        map_grp.attrs["time_offset"] = time_offset

    print(f"Filtered data for '{rec_path}' saved to '{out_file}'")
