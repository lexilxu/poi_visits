import os
import re
import gzip
import gdown
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path("./Data")
RAW_VISITS_DIR = BASE_DIR / "MonthlyPattern" 
RAW_POI_DIR = BASE_DIR / "POIs"
PROCESSED_MONTHLY_DIR = BASE_DIR / "processed"

RAW_VISITS_DIR.mkdir(parents=True, exist_ok=True)
RAW_POI_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)


VISITS_FILES = [
    {"month": "2020-11", "folder_url": "https://drive.google.com/drive/folders/1SUAcuATmJF0VKzzmMAzdAUdPzDn12ShE", "name": "2020-11"},
    {"month": "2020-12", "folder_url": "https://drive.google.com/drive/folders/1JI7r1QCuy1HXPacYFC6UcYB5UnoYuICh", "name": "2020-12"},
    {"month": "2021-01", "folder_url": "https://drive.google.com/drive/folders/1hJwtEieWOaKIRfn-M6U8uwZRExg72Tut", "name": "2021-01"},
]

POI_FILES = [
    {"month": "2020-11", "folder_url": "https://drive.google.com/drive/folders/1Qcwz5IrGvG2QTSJY0juUF1oIu68IWQkc", "name": "2020-11"},
    {"month": "2020-12", "folder_url": "https://drive.google.com/drive/folders/1ExdE9oYgq1BuRr6QqQXhdFsvaeKCXbO5", "name": "2020-12"},
    {"month": "2021-01", "folder_url": "https://drive.google.com/drive/folders/1E-Fn9SL7L8X2oty9dKpbtrVEZdMAk-YH", "name": "2021-01"},
]


def batch_download_gdown(folders, output_dir: Path):
    for folder in folders:
        out_path = output_dir / folder["name"]
        out_path.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            url=folder["folder_url"],
            output=str(out_path),
            quiet=False,
            remaining_ok=True,
        )


batch_download_gdown(VISITS_FILES, RAW_VISITS_DIR)
batch_download_gdown(POI_FILES, RAW_POI_DIR)



