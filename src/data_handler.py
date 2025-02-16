import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


class DataHandler:
    def __init__(
            self,
            input_path: str = 'data',
            output_path: str = 'data/Greek/training/candidate_summaries/'
            ):
        self.input_path = input_path
        self.output_path = output_path

    def _check_data_exists(self, path) -> bool:
        """Checks if the 'data' folder has any items."""
        n_path_items = len(os.listdir(path))
        return n_path_items != 0

    def collect_data(self) -> None:
        """
        Adds 'Greek data' from FNS to the 'data' folder,
        if it doesn't have any other items.
        """
        if not self._check_data_exists(path=self.input_path):
            greek_data_url = 'https://github.com/iit-Demokritos/FNS2023_data/raw/refs/heads/main/Greek.zip'
            http_response = urlopen(greek_data_url)
            zipfile = ZipFile(BytesIO(http_response.read()))
            zipfile.extractall(path='data')
