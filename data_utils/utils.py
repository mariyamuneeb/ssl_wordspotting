import os
import shutil

CWD = os.getcwd()
print(CWD)
IEHR = "/home/mujahid/PycharmProjects/ssl_wordspotting/datasets/IEHR"
IEHR3 = f"{IEHR}/IEHHR_training_part3"
IEHR2 = f"{IEHR}/IEHHR_training_part2"
IEHR1 = f"{IEHR}/IEHHR_training_part1"
IEHRTest = f"{IEHR}/IEHHR_test"
DESTR = "/home/mujahid/PycharmProjects/ssl_wordspotting/datasets/IEHR/full_dataset"
DESTE = "/home/mujahid/PycharmProjects/ssl_wordspotting/datasets/IEHR/full_dataset_test"

IEHRTR = [IEHR1, IEHR2, IEHR3]
IEHRTE = [IEHRTest]


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