"""
"""

import os
import csv
import json

import numpy as np
from ssl_simulator import create_dir

#######################################################################################

class DataLogger:
    def __init__(self, labels, filename):
        self.filename = filename
        self.labels = ["time"] + labels
        self.data = None

        create_dir(os.path.dirname(filename), verbose=False)

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

class RealTimeLogger: # TODO: more testing
    def __init__(self, labels, log_size=10):
        self.labels = ["time"] + labels
        self.log_size = log_size
        self.data = {label: None for label in self.labels}

    def log(self, time, sim_data):
        # Merge time with sim_data
        sim_data = {"time": time, **sim_data}

        for label in self.labels:
            value = sim_data[label]

            # Convert value to a 1D or 2D array (depending on use)
            value = np.atleast_1d(value)

            if self.data[label] is None:
                self.data[label] = value[None, ...]  # Add batch dimension
            else:
                self.data[label] = np.concatenate([self.data[label], value[None, ...]], axis=0)

                # Remove oldest entry if size exceeded
                if len(self.data[label]) > self.log_size:
                    self.data[label] = self.data[label][-self.log_size:]

#######################################################################################