"""
This module provides collects Greek annual reports.

It includes functions to check whether data already exists
and if it doesn't, it downloads it and unzips it.
"""


import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


class DataCollector:
    def __init__(
            self,
            input_path: str = 'data',
            output_path: str = 'data/Greek/training/candidate_summaries/'
            ):
        """Collects Greek data.

        Args:
            input_path (str, optional): The path to retrieve data from. Defaults to 'data'.
            output_path (str, optional): The path to save the collected data. Defaults to 'data/Greek/training/candidate_summaries/'.

        """
        self.input_path = input_path
        self.output_path = output_path

    def _check_data_exists(self, path) -> bool:
        """Checks if the 'data' folder has any items.

        Args:
            path (str): The path to be checked for existing data.

        Returns:
            bool: True if data already exists and False if it doesn't.

        """
        n_path_items = len(os.listdir(path))
        return n_path_items != 0

    def collect_data(self) -> None:
        """Adds 'Greek data' from FNS to the 'data' folder."""

        if not self._check_data_exists(path=self.input_path):
            greek_data_url = 'https://github.com/iit-Demokritos/FNS2023_data/raw/refs/heads/main/Greek.zip'
            http_response = urlopen(greek_data_url)
            zipfile = ZipFile(BytesIO(http_response.read()))
            zipfile.extractall(path='data')
