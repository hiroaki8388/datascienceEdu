import os
import shutil

from google.cloud import storage
from google.cloud.storage import Blob



def download(YEAR, MONTH, dstdir):
    logging.info(f'requseting data for {YEAR},{MONTH}')

