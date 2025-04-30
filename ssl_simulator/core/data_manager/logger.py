"""
"""

__all__ = ["DataLogger"]

import os
import csv
import json

import numpy as np
import pandas as pd
from ..utils import createDir

#######################################################################################

class DataLogger:
    def __init__(self, labels, filename):
        self.filename = filename
        self.labels = ["time"] + labels
        createDir(os.path.dirname(filename), verbose=False)

        # Create file and write header
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.labels)
            writer.writeheader()

    def log(self, time, sim_data):
        row = {"time": time}
        
        # Convert arrays to JSON strings
        for key, value in sim_data.items():
            if isinstance(value, np.ndarray):
                row[key] = json.dumps(value.tolist())
            else:
                row[key] = value

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.labels)
            writer.writerow(row)

#######################################################################################