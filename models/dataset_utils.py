from torch.utils.data import Dataset
import os
from PIL import Image
import random


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


class IAMDataset(BaseDataset):
    """
    train_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/datasets/IEHR/words_training'
    test_root_dir = '/home/mujahid/PycharmProjects/ssl_wordspotting/datasets/IEHR/words_test'
    train_dataset = IAMDataset(train_root_dir)
    test_dataset = IAMDataset(test_root_dir)
    """
    pass


if __name__ == "__main__":
    pass
