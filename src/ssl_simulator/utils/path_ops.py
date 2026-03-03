import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def create_dir(directory: str) -> None:
    """
    Create a new directory if it doesn't already exist.

    Args:
        directory (str): The path of the directory to create.
    """
    try:
        os.mkdir(directory)
        logger.debug(f"Directory '{directory}' created!")
    except FileExistsError:
        logger.debug(f"The directory '{directory}' already exists!")


def add_src_to_path(file=None, relative_path="", deep=0):
    """
    Adds the "relative_path" folder to sys.path based on 'file' or actual location.
    """
    root = Path(file).resolve().parents[deep + 1] if file else Path.cwd().parents[deep]
    target = root / relative_path
    sys.path.append(str(target))

    logger.debug(f"Added to sys.path: root={root}, target={target}")
