"""
"""

__all__ = ["GvfLineHeading"]

import numpy as np

from .gvf_line import GvfLine

#######################################################################################

class GvfLineHeading(GvfLine):
  def __init__(self, A, heading, line_length=30):
    # Save point A as numpy array and heading
    self.A = np.array(A)
    self.heading = heading

    # Calculate the slope (m) using the heading angle
    m = np.sin(heading) / np.cos(heading)

    # The y-intercept (b) is the y-coordinate of point A
    b = self.A[1]

    # Call the parent class constructor
    super().__init__(m = m, b = b, line_length = line_length)

#######################################################################################