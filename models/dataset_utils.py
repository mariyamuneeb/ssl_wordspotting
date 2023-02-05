from torch.utils.data import Dataset
import os
# from PIL import Image
# import random
# import pathlib
import logging
from paths import IAM_XML_DIR, IAM_DIR, IAM_RULE_DIR, IAM_WORDS_DIR
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.DEBUG)


# class BaseDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         super(BaseDataset, self).__init__()
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_paths = self._get_image_paths()
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         img_name = self.img_paths[idx]
#         image = Image.open(img_name)
#         if self.transform:
#             image = self.transform(image)
#         return image, 'na'
#
#     def _get_image_paths(self):
#         return [os.path.join(self.root_dir, i)
#                 for i in os.listdir(self.root_dir)]
#
#     def get_random_samples(self, number=9):
#         random_imgs = random.sample(self.img_paths, number)
#         random_imgs = [Image.open(i) for i in random_imgs]
#         return random_imgs
#
#
# class IAMDataset(BaseDataset):
#     """
#     train_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/datasets/IEHR/words_training'
#     test_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/datasets/IEHR/words_test'
#     train_dataset = IAMDataset(train_root_dir)
#     test_dataset = IAMDataset(test_root_dir)
#     """
#     pass


class IAMDataset2(Dataset):
    def __init__(self, ttype, transform=None):
        self.ttype = ttype
        if ttype == 'train':
            self.rule_file_path = IAM_RULE_DIR / "trainset.txt"
        elif ttype == 'test':
            self.rule_file_path = IAM_RULE_DIR / "testset.txt"
        elif ttype == 'val':
            self.rule_file_path = IAM_RULE_DIR / "validationset1.txt"

    # def _xml_file_parsing(self):
    #     return transcripts
    def get_xml_file_object(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        return root

    def create_samples(self):
        dir_paths = self.create_dir_paths()
        samples = list()
        for sample_name, dir_path in dir_paths:
            print(sample_name,dir_path)
            image_names = list(dir_path.glob('*.png'))
            xml_file_name = f'{dir_path.name}.xml'
            xml_file_path = IAM_XML_DIR / xml_file_name
            root = self.get_xml_file_object(xml_file_path)
            filtered_image_paths = [i for i in image_names if sample_name in i.name]
            filtered_image_names = [i.name for i in filtered_image_paths]
            image_names_0 = [i.split('.')[0] for i in filtered_image_names]
            for word in root.iter('word'):
                text = word.get('text')
                id = word.get('id')
                idx = image_names_0.index(id)
                samples.append((filtered_image_paths[idx], text))
                print(samples)
        return samples

    def create_xml_paths(self):
        dir_names = self.sample_names()
        xml_names = [i.split('-')[:-1] for i in dir_names]
        xml_file_names = list(dict.fromkeys(['-'.join(map(str, i)) + '.xml' for i in xml_names]))
        xml_file_paths = [IAM_XML_DIR / i for i in xml_file_names]
        return xml_file_paths

    def create_dir_paths(self):
        sample_names = self.sample_names()
        dir_paths = list()
        for sample_name in sample_names:
            splits = sample_name.split('-')
            l1 = splits[0]
            l2 = '-'.join(splits[:-1])
            dir_path = IAM_WORDS_DIR / l1 / l2
            dir_paths.append((sample_name, dir_path))
        logging.info(f"{len(dir_paths)} dirs for {self.ttype} set")
        # print(dir_paths)
        return dir_paths

    def sample_names(self):
        with open(self.rule_file_path) as f:
            dir_names = f.readlines()
            dir_names = [i.replace('\n', '').strip() for i in dir_names]
        logging.info(f"{len(dir_names)} dirs for {self.ttype} set")
        return dir_names

    def image_names(self, dir_path):
        image_names = os.listdir(dir_path)
        return image_names


if __name__ == "__main__":
    lms = IAMDataset2(ttype='test').create_xml_paths()
    ims = IAMDataset2(ttype='test').create_dir_paths()
    sms = IAMDataset2(ttype='test').create_samples()
    print(len(sms))

    # print(lms[:10])
    # print(ims[:20])
