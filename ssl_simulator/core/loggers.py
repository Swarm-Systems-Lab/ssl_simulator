"""
"""

import os
import csv

import numpy as np
from ssl_simulator import create_dir, dict_to_json

#######################################################################################

class DataLogger:
    def __init__(self, labels, filename, log_size=None, settings={}):
        """
        Flexible logger for in-memory and/or file-based logging.

        Parameters
        ----------
        labels : list of str
            Labels for data columns (excluding "time").
        filename : str or None
            Path to CSV file for persistent logging. If None, only in-memory logging is used.
        log_size : int or None
            Maximum number of time steps to keep in memory. If None, full history is stored.
        settings : dict
            Dictionary of settings to store in the file header (if filename is used).
        """
        self.filename = filename
        self.labels = ["time"] + labels
        self.log_size = log_size
        self.data = {label: None for label in self.labels}
        self.settings = settings

        if self.filename:
            create_dir(os.path.dirname(filename), verbose=False)

            with open(self.filename, mode='w', newline='') as file:
                file.write(f"# SETTINGS: {dict_to_json(self.settings, dump=True)}\n")
                writer = csv.DictWriter(file, fieldnames=self.labels)
                writer.writeheader()

    def log(self, time, sim_data):
        """
        Log one time step of data.

        Parameters
        ----------
        time : float
            Current simulation time.
        sim_data : dict
            Dictionary of data keyed by label.
        """
        sim_data = {"time": time, **sim_data}

        # In-memory logging
        for label in self.labels:
            value = np.atleast_1d(sim_data[label])
            if self.data[label] is None:
                self.data[label] = value[None, ...]
            else:
                self.data[label] = np.concatenate([self.data[label], value[None, ...]], axis=0)
                if self.log_size and len(self.data[label]) > self.log_size:
                    self.data[label] = self.data[label][-self.log_size:]

                    # Remove oldest entry if size exceeded
                    if len(self.data[label]) > self.log_size:
                        self.data[label] = self.data[label][-self.log_size:]

        # File logging
        if self.filename:
            row = dict_to_json(sim_data)
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.labels)
                writer.writerow(row)

#######################################################################################