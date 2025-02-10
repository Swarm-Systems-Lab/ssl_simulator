"""
"""

__all__ = ["GvfLineAB"]

import numpy as np

from .gvf_line_heading import GvfLineHeading

#######################################################################################

class GvfLineAB(GvfLineHeading):
  def __init__(self, A, B):
    # Save points as NumPy arrays
    self.A = np.array(A)
    self.B = np.array(B)

    # Calculate the heading of the line, which is related to the slope (m = dy/dx)
    dx = B[0] - A[0]
    dy = B[1] - A[1]
    heading = np.arctan2(dy,dx)

    # Compute the Euclidean distance between A and B (line length)
    AB = self.B - self.A
    line_length = np.linalg.norm(AB)

    # Call the parent class constructor
    super().__init__(A=self.A, heading=heading, line_length=line_length)

#######################################################################################