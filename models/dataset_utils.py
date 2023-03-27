import pathlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
# import pathlib
import logging
from paths import IAM_XML_DIR, IAM_DIR, IAM_RULE_DIR, IAM_WORDS_DIR
from xml.etree import ElementTree as ET
from datetime import datetime
import numpy as np
import glob

logging.basicConfig(level=logging.DEBUG)


class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(BaseDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = self._get_image_paths()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 'na'

    def _get_image_paths(self):
        return [os.path.join(self.root_dir, i)
                for i in os.listdir(self.root_dir)]

    def get_random_samples(self, number=9):
        random_imgs = random.sample(self.img_paths, number)
        random_imgs = [Image.open(i) for i in random_imgs]
        return random_imgs


#
class IAMDataset(BaseDataset):
    """
    train_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/data/IEHR/words_training'
    test_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/data/IEHR/words_test'
    train_dataset = IAMDataset(train_root_dir)
    test_dataset = IAMDataset(test_root_dir)
    """
    pass


## images : words/l1/l1-l2/l1-l2-ldx-idx.png : sample
## l1 -  sample_group_root_dir
## l1-l2 - sample_group_dir
## rules : l1-l2-ldx - sample_sub_group
## xml : l1-l2-ldx-idx - sample_name


class IAMDataset2(Dataset):
    def __init__(self, ttype, transform=None):
        self.transform = transform
        self.ttype = ttype
        if ttype == 'train':
            self.rule_file_path = IAM_RULE_DIR / "trainset.txt"
        elif ttype == 'test':
            self.rule_file_path = IAM_RULE_DIR / "testset.txt"
        elif ttype == 'val':
            self.rule_file_path = IAM_RULE_DIR / "validationset1.txt"
        self.line_folders = None
        self.line_folders, self.line_dirs = self.create_line_dirs()
        self.samples = self.get_word_transcripts()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, img_path, transcript = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img_id, img, transcript

    def get_xml_file_object(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        return root

    def get_word_transcripts(self):
        word_ids, word_paths = self.get_words()

        word_ids, xml_paths = self.construct_xml_file_paths(word_ids)
        ll = list()
        for word_id, word_path, xml_path in zip(word_ids, word_paths, xml_paths):
            root = self.get_xml_file_object(xml_path)
            for word in root.iter('word'):
                img_id = word.get('id')
                if img_id == word_id:
                    transcript = word.get('text')
                    ll.append((word_id, word_path, transcript))
        return ll

    def construct_xml_file_paths(self, word_ids):
        xml_paths = ['-'.join(i.split('-')[:-2]) + '.xml' for i in word_ids]
        xml_paths = [IAM_XML_DIR / i for i in xml_paths]
        return word_ids, xml_paths

    def get_words(self):
        # print(len(sample_group_dir))

        image_paths = [glob.glob(f"{i}/*.png") for i in self.line_dirs]
        # print(sample_file_paths)

        image_paths = [item for sublist in image_paths for item in sublist]

        line_ids = self.read_line_ids()

        word_paths = [i for i in image_paths if '-'.join(pathlib.Path(i).name.split('-')[:-1])
                      in line_ids]
        word_ids = [pathlib.Path(i).name.split('.')[0] for i in word_paths]

        return word_ids, word_paths
        #

    def create_line_dirs(self):
        # images : words/l1/l1-l2/l1-l2-ldx-idx.png : sample
        # l1 -  sample_group_root_dir
        # l1-l2 - line_folders
        # rules : l1-l2-ldx - line_ids
        # xml file name: l1-l2.xml
        # xml : l1-l2-ldx-idx - sample_name
        line_ids = self.read_line_ids()
        line_folders = [f"{i.split('-')[0]}-{i.split('-')[1]}" for i in line_ids]
        line_folders = list(dict.fromkeys(line_folders))
        line_dirs = [IAM_WORDS_DIR / i.split('-')[0] / f"{i.split('-')[0]}-{i.split('-')[1]}"
                     for i in line_ids]
        line_dirs = list(dict.fromkeys(line_dirs))
        return line_folders, line_dirs

    def read_line_ids(self):
        # this method reads the rules.txt files and
        # returns its contents sample_sub_groups
        with open(self.rule_file_path) as f:
            line_ids = [i.replace('\n', '').strip() for i in f.readlines()]
        logging.info(f"{len(line_ids)} dirs for {self.ttype} set")
        return line_ids

    def image_names(self, dir_path):
        image_names = os.listdir(dir_path)
        return image_names

    def get_random_samples(self, number=9):
        random_samples = random.sample(self.samples, number)
        random_samples = [(i[0], Image.open(i[0]), i[2]) for i in random_samples]
        return random_samples


if __name__ == "__main__":
    dataset = IAMDataset2(ttype='test')
    x, y,z = dataset[1]
    print(x)
    print(type(y))
    print(z)

