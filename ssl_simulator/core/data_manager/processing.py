"""
"""

__all__ = ["load_sim"]

import pandas as pd
import json
import numpy as np

#######################################################################################

def load_sim(filename, debug=False):
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