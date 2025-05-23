import site

# this both inserts the directory *and* processes any .pth files in it
site.addsitedir(
    "/home/lq53/miniconda3/envs/neuroposelib/lib/python3.9/site-packages"
)

# now Python has read __editable__.neuroposelib-0.2.dev0.pth
# and added your repo path automatically.
import neuroposelib
print("✅ loaded from:", neuroposelib.__file__)



import json
import h5py
import numpy as np
import pandas as pd
import os
import pickle  # used for pickling non-uniform arrays

def export_aligned_data_to_h5(
    data_obj, 
    rec_path, 
    frame_mapping_file, 
    out_file
):
    """
    Filters data_obj to a specific rec_path and frames from frame_mapping_file,
    then saves minimal data to an HDF5 file.
    
    This version preserves the shape of any np.ndarray stored in a DataFrame column.
    For columns with ndarray entries:
      - If all arrays have the same shape, they are stacked and saved directly.
      - If they vary in shape, each array is pickled and stored as a binary blob.
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
    
    # 3. Filter rows by mapped frame indices (assuming your DataFrame has a 'frame' column)
    if "frame" not in path_data.columns:
        raise ValueError("DataFrame does not have 'frame' column to filter by.")
    
    # Adjust frames so that they start at 0
    min_frame = path_data["frame"].min()
    path_data["frame"] = path_data["frame"] - min_frame

    # Create helper offsets DataFrame
    offsets = pd.DataFrame({'offset': range(10)})

    # Cross-merge (pandas 1.2+ supports `how="cross"`)
    expanded = path_data.merge(offsets, how='cross')

    # Update the frame by adding the offset
    expanded['frame'] = expanded['frame'] + expanded['offset']
    expanded.drop(columns='offset', inplace=True)
    
    filtered_data = expanded[expanded["frame"].isin(mapped_frames)]
    
    if filtered_data.empty:
        print("No overlapping frames found between path_data and mapped_sixcam_frame_indices.")
    
    # 4. Save to HDF5
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with h5py.File(out_file, "w") as hf:
        
        # (A) Save filtered DataFrame columns
        grp = hf.create_group("filtered_data")
        
        for col in filtered_data.columns:
            col_data = filtered_data[col].to_numpy()
            
            # If the first element is an ndarray, handle it specially.
            if len(col_data) > 0 and isinstance(col_data[0], np.ndarray):
                try:
                    # Try stacking arrays if they have the same shape
                    stacked = np.stack(col_data)
                    grp.create_dataset(col, data=stacked, compression="gzip")
                except ValueError:
                    # For non-uniform shapes, pickle each array so that its structure is preserved.
                    binary_dtype = h5py.special_dtype(vlen=bytes)
                    pickled_data = np.array([pickle.dumps(item) for item in col_data])
                    grp.create_dataset(col, data=pickled_data, dtype=binary_dtype, compression="gzip")
            
            # Handle string/text columns
            elif col_data.dtype.kind in ["O", "U", "S"]:
                variable_length_string_dt = h5py.special_dtype(vlen=str)
                # Convert each element explicitly to a Python string to avoid fixed-length unicode issues.
                string_list = [str(x) for x in col_data]
                grp.create_dataset(
                    col, 
                    data=np.array(string_list, dtype=object), 
                    dtype=variable_length_string_dt, 
                    compression="gzip"
                )
            else:
                # For numeric or other dtypes, write directly
                grp.create_dataset(col, data=col_data, compression="gzip")
        
        # (B) Save relevant data_obj attributes
        if hasattr(data_obj, "embed_vals") and data_obj.embed_vals is not None:
            hf.create_dataset("embed_vals", data=data_obj.embed_vals, compression="gzip")
        
        # Example: saving meta info
        if hasattr(data_obj, "meta") and data_obj.meta is not None:
            meta_grp = hf.create_group("meta")
            if isinstance(data_obj.meta, dict):
                for key, val in data_obj.meta.items():
                    val_array = np.array(val, dtype=object)
                    if val_array.dtype.kind in ["O", "U", "S"]:
                        variable_length_string_dt = h5py.special_dtype(vlen=str)
                        meta_grp.create_dataset(
                            key,
                            data=np.array([str(x) for x in val_array.flatten()], dtype=object),
                            dtype=variable_length_string_dt,
                            compression="gzip"
                        )
                    else:
                        meta_grp.create_dataset(key, data=val_array, compression="gzip")
            else:
                meta_vals = np.array(data_obj.meta, dtype=object)
                if meta_vals.dtype.kind in ["O", "U", "S"]:
                    variable_length_string_dt = h5py.special_dtype(vlen=str)
                    meta_vals = np.array([str(x) for x in meta_vals.flatten()], dtype=object)
                    meta_grp.create_dataset(
                        "meta_data",
                        data=meta_vals,
                        dtype=variable_length_string_dt,
                        compression="gzip"
                    )
                else:
                    meta_grp.create_dataset("meta_data", data=meta_vals, compression="gzip")
        
        # (C) Save frame mapping info
        map_grp = hf.create_group("frame_mapping")
        map_grp.create_dataset(
            "mapped_sixcam_frame_indices",
            data=np.array(map_data["mapped_sixcam_frame_indices"]),
            compression="gzip"
        )
        map_grp.attrs["time_offset"] = time_offset

    print(f"Filtered data for '{rec_path}' saved to '{out_file}'")





def export_aligned_data_from_paths(
    pickle_path,
    rec_path,
    frame_mapping_file,
    out_file
):
    """
    Loads the pickled data_obj and then calls export_aligned_data_to_h5().
    Usage example:
        export_aligned_data_from_paths(
            "/path/to/datastruct.p",
            rec_path,
            frame_mapping_file,
            out_file
        )
    """
    with open(pickle_path, "rb") as f:
        data_obj = pickle.load(f)
    export_aligned_data_to_h5(
        data_obj=data_obj,
        rec_path=rec_path,
        frame_mapping_file=frame_mapping_file,
        out_file=out_file
    )


def load_new_filtered_data_from_h5(h5_file_path):
    """
    Loads the 'filtered_data' group from an HDF5 file into a Pandas DataFrame,
    handling pickled numpy arrays (used when inner shapes were non-uniform)
    as well as string datasets and numeric data.
    """


    with h5py.File(h5_file_path, "r") as hf:
        grp = hf["filtered_data"]
        data_dict = {}
        dfs_list = []
        for col_name in grp.keys():
            ds = grp[col_name]
            col_data = ds[:]
            
            # # If the dataset was stored with pickled objects, unpickle each element.
            # if ds.attrs.get("pickled", False):
            #     col_data = np.array([pickle.loads(elem) for elem in col_data])
            # # Otherwise, if it's a string dataset, decode bytes if necessary.
            # elif col_data.dtype.kind in ["S", "O"]:
            #     # Check if the array is not empty and first element is bytes.
            #     if col_data.size > 0 and isinstance(col_data.flat[0], bytes):
            #         col_data = np.array([elem.decode("utf-8") if isinstance(elem, bytes) else elem 
            #                               for elem in col_data])
            # Otherwise, leave the data as is.
            if col_data.ndim > 1:
                expanded = pd.DataFrame([m.flatten() for m in col_data])
                expanded.columns = [f'{col_name}_{i}' for i in range(expanded.shape[1])]
                dfs_list.append(expanded)
            else:
                data_dict[col_name] = col_data
        
    df = pd.concat([pd.DataFrame(data_dict), *dfs_list], axis=1)
    return df




def batch_export_and_csv(
    post_ana, 
    base_folder, 
    pickle_path
):
    """
    For each row in post_ana.to_pandas(), build:
      rec_path = base_folder / date_folder / rec_file
    then:
      - ensure rec_path/MIR_Aligned exists
      - call export_aligned_data_from_paths(pickle_path, rec_path, frame_map, out_h5)
      - load back the filtered_data group into a DataFrame
      - save that DataFrame to rec_path/MIR_Aligned/filtered_data.csv
    """
    # 1. turn post_ana into pandas and build rec paths
    df = post_ana.to_pandas()
    df['rec_path'] = df.apply(
        lambda row: os.path.join(base_folder, row['date_folder'], row['rec_file']),
        axis=1
    )
    
    for _, row in df.iterrows():
        rec_path = row['rec_path']
        mir_dir = os.path.join(rec_path, "MIR_Aligned")
        os.makedirs(mir_dir, exist_ok=True)
        
        # 2. frame mapping JSON is always in MIR_Aligned/frame_mapping.json
        frame_map = os.path.join(mir_dir, "frame_mapping.json")
        
        # 3. name the HDF5 and CSV outputs
        out_h5  = os.path.join(mir_dir, "filtered_data.h5")
        out_csv = os.path.join(mir_dir, "filtered_data.csv")
        
        print(f"\n--- Processing {rec_path} ---")
        # 4. export the HDF5
        export_aligned_data_from_paths(
            pickle_path,
            rec_path,
            frame_map,
            out_h5
        )
        
        # 5. load it back in and dump CSV
        try:
            df_filt = load_new_filtered_data_from_h5(out_h5)
            df_filt.to_csv(out_csv, index=False)
            print(f" → CSV saved to: {out_csv}")
        except Exception as e:
            print(f" ⚠️  Failed to load/write CSV for {rec_path}: {e}")
