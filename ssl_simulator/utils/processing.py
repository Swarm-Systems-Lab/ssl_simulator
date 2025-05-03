"""
"""

__all__ = [
    "load_sim",
]

import os
import json

import pandas as pd
import numpy as np

#######################################################################################

def load_sim(filename, debug=False, max_size_mb=100):
    # Check file size in bytes
    file_size = os.path.getsize(filename)
    if file_size > max_size_mb * 1024 * 1024:
        raise ValueError(
            f"File size is {file_size / (1024 * 1024):.2f} MB, which exceeds the limit of {max_size_mb} MB. "
            f"You can increase this limit by setting the 'max_size_mb' parameter."
        )
    
    df = pd.read_csv(filename)
    labels = df.columns.drop("time").tolist()

    data_dict = {"time": df["time"].to_numpy()}
    for col in labels:
        try:
            # Parse JSON strings into arrays
            parsed_col = df[col].apply(lambda x: np.array(json.loads(x)))
            data_dict[col] = np.stack(parsed_col.to_numpy())
        except Exception:
            # If it's a flat value
            data_dict[col] = df[col].to_numpy()

    if debug:
        for key, val in data_dict.items():
            print(f"{key}: shape {val.shape}")

    return data_dict

#######################################################################################