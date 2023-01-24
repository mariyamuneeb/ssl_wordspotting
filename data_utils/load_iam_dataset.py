import shutil
from google.colab import drive
import os
import zipfile
drive.mount('/content/drive')
GDRIVE_ROOT = '/content/drive/MyDrive/Datasets'

dataset_zip= f'{GDRIVE_ROOT}/IAM_HW/words_full_dataset.zip'
dest_zip = '/content/ssl_wordspotting/words_full_dataset.zip'
shutil.copy(dataset_zip, dest_zip)
os.mkdir('/content/ssl_wordspotting/data')

with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
    zip_ref.extractall('/content/ssl_wordspotting/data')