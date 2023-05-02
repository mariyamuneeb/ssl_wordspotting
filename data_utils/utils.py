import shutil
import os
import tarfile
import zipfile
from google.colab import drive
from pathlib import Path

CWD = os.getcwd()
print(CWD)
IEHR = "/home/mujahid/PycharmProjects/ssl_wordspotting/data/IEHR"
IEHR3 = f"{IEHR}/IEHHR_training_part3"
IEHR2 = f"{IEHR}/IEHHR_training_part2"
IEHR1 = f"{IEHR}/IEHHR_training_part1"
IEHRTest = f"{IEHR}/IEHHR_test"
DESTR = "/home/mujahid/PycharmProjects/ssl_wordspotting/data/IEHR/full_dataset"
DESTE = "/home/mujahid/PycharmProjects/ssl_wordspotting/data/IEHR/full_dataset_test"

IEHRTR = [IEHR1, IEHR2, IEHR3]
IEHRTE = [IEHRTest]


def connect_to_gdrive():
    # connects to gdrive and returns root of gdrive
    drive.mount('/content/drive')
    GDRIVE_ROOT = '/content/drive/MyDrive'
    return GDRIVE_ROOT


def copy_iam_dataset_to_colab():
    FILES_TO_COPY = ['words.tgz', "xml.tgz", "rules.zip"]
    print("Copying IAM Dataset from GDrive")
    GDRIVE_ROOT = connect_to_gdrive()
    GDRIVE_DATA_ROOT = Path(f"{GDRIVE_ROOT}/Datasets")
    IAM_DIR = GDRIVE_DATA_ROOT / 'IAM_HW'
    DEST_DATA_ROOT = Path("/content/ssl_wordspotting/data")
    DEST_IAM_HW = DEST_DATA_ROOT / "IAM_HW"

    os.mkdir(DEST_DATA_ROOT)
    os.mkdir(DEST_IAM_HW)
    SOURCE_FILES = [IAM_DIR / file for file in FILES_TO_COPY]
    DEST_FILES = [DEST_IAM_HW / file for file in FILES_TO_COPY]
    for s, d in zip(SOURCE_FILES, DEST_FILES):
        shutil.copy(s, d)
        if not os.path.isfile(d):
            raise FileNotFoundError(f"{d} file not found")
        else:
            print(f"{d} now copied")
        if '.tgz' in d.name:
            with tarfile.open(d) as archive_ref:
                archive_ref.extractall(DEST_IAM_HW / d.name.split('.')[0])
        else:
            with zipfile.ZipFile(d, 'r') as zip_ref:
                zip_ref.extractall(DEST_IAM_HW)
    print("Copied IAM HW Files")


def copy_iehr_dataset_to_colab():
    ## copy IEHR Dataset from Gdrive to colab
    print("Copying IEHR Dataset from GDrive")
    GDRIVE_ROOT = connect_to_gdrive()
    DATASET_ROOT = f"{GDRIVE_ROOT}/Datasets"
    dataset_zip = f'{DATASET_ROOT}/IEHR/words_full_dataset.zip'
    dest_zip = '/content/ssl_wordspotting/words_full_dataset.zip'
    shutil.copy(dataset_zip, dest_zip)
    if not os.path.isfile(dest_zip):
        raise FileNotFoundError(f"{dest_zip} file not found")
    else:
        print("IAM Dataset Zip now copied")
    os.mkdir('/content/ssl_wordspotting/iam_data')
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall('/content/ssl_wordspotting/data')


def get_images_paths(paths, dest):
    sumt = list()
    sumj = 0
    for iehr_path in paths:
        records = os.listdir(iehr_path)
        records_path = [f"{iehr_path}/{i}" for i in records]
        sumi = 0
        for record_path in records_path:
            words_path = f"{record_path}/words"
            images = [i for i in os.listdir(words_path) if ".png" in i]
            sumi += len(images)
            for img in images:
                dimg = f"{dest}/{img}"
                simg = f"{words_path}/{img}"
                shutil.copy(simg, dimg)
                if os.path.isfile(dimg):
                    sumj += 1
        sumt.append(sumi)
    print(sumt)
    print(sum(sumt))
    print(f"copied {sumj} images")
