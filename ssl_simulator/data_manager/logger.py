"""
"""

__all__ = ["DataLogger"]

import os

import pandas as pd
from ..utils import createDir

#######################################################################################

class DataLogger:
    def __init__(self, labels):
        self.data = {"time": []}
        for key in labels:
            self.data.update({key: []})

    def log(self, time, sim_data):
        self.data["time"].append(time)
        for key, value in sim_data.items():
            self.data[key].append(value)
        
    def save(self, filename):
        createDir(os.path.dirname(filename), verbose=False)
        df = pd.DataFrame(self.data)
        df.to_pickle(filename)

#######################################################################################