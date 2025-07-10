import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from torch.utils.data import Dataset
from PIL import Image
import os
from itertools import islice

class DRDataset(Dataset):
    def __init__(self, base_path, data_file, transform=None):
        self.transform = transform
        self.data_list = self.get_files(base_path, data_file=data_file)
    def __len__(self):
        return len(self.data_list)
    def get_files(self,root, data_file):
        import csv
        csv_reader = csv.reader(open(data_file))
        img_list = []
        for line in islice(csv_reader, 1, None):
            img_list.append(
                [
                   os.path.join(root,line[0]),
                    int(line[1])
                ]
            )
        return img_list

    def __getitem__(self, idx):
        img_path,label = self.data_list[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

