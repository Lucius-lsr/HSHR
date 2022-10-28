import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class PathologyDataset(Dataset):
    def __init__(self, root, transform, num_patch=100) -> None:
        super().__init__()
        dirs = os.listdir(root)
        self.size = num_patch * len(dirs)
        self.root = root
        self.transform = transform
        self.num_patch = num_patch

    def __getitem__(self, idx: int):
        file = os.path.join(self.root, 'slide_{}'.format(int(idx / self.num_patch)), '{}.jpg'.format(idx % self.num_patch))
        img = Image.open(file)
        ret = self.transform(img)
        return ret, np.zeros([1])

    def __len__(self) -> int:
        return self.size
