# -*- coding: utf-8 -*-
# Time: 2023/6/20 12:02
import os


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    os.makedirs(path, exist_ok=True)
