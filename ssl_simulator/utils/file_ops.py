"""
"""

__all__ = [
    "check_file_size",
]

import os

#######################################################################################

def check_file_size(filename, max_size_mb=None):
    file_size = os.path.getsize(filename)
    if max_size_mb:
        if file_size > max_size_mb * 1024 * 1024:
            raise ValueError(
                f"File size is {file_size / (1024 * 1024):.2f} MB, which exceeds the limit of {max_size_mb} MB. "
                f"You can increase this limit by setting the 'max_size_mb' parameter."
            )
#######################################################################################