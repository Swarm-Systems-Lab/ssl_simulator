import csv
import os
from typing import Any

import numpy as np

from ssl_simulator.utils.dict_ops import dict_to_json
from ssl_simulator.utils.path_ops import create_dir

#######################################################################################

_INITIAL_CAPACITY = 256


class DataLogger:
    def __init__(self, labels, filename, log_size=None, settings=None):
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
        if settings is None:
            settings = {}
        self.filename: str | None = filename
        self.labels: list[str] = ["time", *labels]
        self.log_size: int | None = log_size
        self.data: dict[str, np.ndarray | None] = dict.fromkeys(self.labels)
        self.settings: dict[str, Any] = settings

        # --- Internal buffer state ---
        # _raw: pre-allocated numpy array per label (shape: [capacity, *value_shape])
        # _fill: total number of entries written (monotonically increasing)
        # _capacity: current allocated capacity
        self._raw: dict[str, np.ndarray] = {}
        self._fill: int = 0
        if log_size is not None:
            # For bounded logging: allocate exactly log_size slots (ring buffer)
            self._capacity: int = log_size
        else:
            # For unbounded logging: start small, double as needed
            self._capacity = _INITIAL_CAPACITY

        # --- File logging: open once, keep open for simulation lifetime ---
        self._csv_file = None
        self._csv_writer = None
        if self.filename:
            create_dir(os.path.dirname(filename), verbose=False)
            with open(self.filename, mode="w", newline="") as file:
                file.write(f"# SETTINGS: {dict_to_json(self.settings, dump=True)}\n")
                csv.DictWriter(file, fieldnames=self.labels).writeheader()
            # Re-open in append mode and keep the handle alive
            self._csv_file = open(self.filename, mode="a", newline="")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.labels)

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

        # --- Initialise raw buffers on the first call (shapes are unknown until runtime) ---
        if self._fill == 0:
            for label in self.labels:
                value = np.atleast_1d(sim_data[label])
                self._raw[label] = np.empty((self._capacity, *value.shape), dtype=value.dtype)

        # --- Grow buffer when full (unbounded case only) ---
        if self.log_size is None and self._fill >= self._capacity:
            self._capacity *= 2
            for label in self.labels:
                old = self._raw[label]
                new_buf = np.empty((self._capacity, *old.shape[1:]), dtype=old.dtype)
                new_buf[: self._fill] = old
                self._raw[label] = new_buf

        # --- Write at the correct index ---
        write_idx = self._fill % self._capacity  # works for both ring and linear buffers
        for label in self.labels:
            self._raw[label][write_idx] = np.atleast_1d(sim_data[label])

        self._fill += 1

        # --- Update self.data views (cheap O(labels) slice / concatenate) ---
        if self.log_size is None:
            # Unbounded: simple prefix slice (numpy view, O(1))
            for label in self.labels:
                self.data[label] = self._raw[label][: self._fill]
        else:
            fill = min(self._fill, self._capacity)
            if self._fill <= self._capacity:
                # Ring not yet wrapped: plain prefix slice
                for label in self.labels:
                    self.data[label] = self._raw[label][:fill]
            else:
                # Ring wrapped: reconstruct chronological order
                # head points to the oldest entry
                head = self._fill % self._capacity
                for label in self.labels:
                    buf = self._raw[label]
                    self.data[label] = np.concatenate([buf[head:], buf[:head]], axis=0)

        # --- File logging (single open file handle, no per-step open/close) ---
        if self._csv_writer is not None:
            row = dict_to_json(sim_data)
            self._csv_writer.writerow(row)

    def close(self) -> None:
        """Flush and close the CSV file if open. Safe to call multiple times."""
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except OSError:
                pass
            finally:
                self._csv_file = None
                self._csv_writer = None

    def __del__(self) -> None:
        self.close()


#######################################################################################
