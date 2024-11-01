# src/components/data_ingestion.py
import os
import urllib.request as request
from box import ConfigBox
from pathlib import Path
from src import logger

def download_file(config: ConfigBox):
    """Download a file from the source URL to the local data file path."""
    if not os.path.exists(config.local_data_file):
        filename, headers = request.urlretrieve(
            url=config.source_URL,
            filename=config.local_data_file
        )
        logger.info(f"{filename} downloaded! with the following info: \n{headers}")
    else:
        logger.info(f"File already exists of size: {get_size(Path(config.local_data_file))}")

def get_size(path: Path) -> str:
    """Get size of the file in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


