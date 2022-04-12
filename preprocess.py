# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/11 11:27
@Author  : Lucius
@FileName: preprocess.py
@Software: PyCharm
"""

import openslide
from torchvision.models import densenet121
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.data_utils import *
from utils.model.base_cnns import ResNetFeature, VGGFeature
import os
import pickle
from utils.sampling import sample_patch_coors
import numpy as np

SVS_DIR = 'YOUR/SVS/FILES/ROOT/DIRECTORY'
RESULT_DIR = 'THE/ROOT/DIRECTORY/OF/YOUR/PREPROCESSED/RESULT'
TMP = 'THE/PATH/OF/TMP/DIRECTORY'



def extract_ft(slide, patch_coors, depth=34, batch_size=128, cnn_base='resnet'):
    if cnn_base == 'resnet':
        model_ft = ResNetFeature(depth=depth, pooling=True, pretrained=True)
        input_img_size = 224
    elif cnn_base == 'densenet':
        densenet = densenet121(pretrained=True)
        densenet = torch.nn.Sequential(*list(densenet.children())[:-1], torch.nn.AvgPool2d(kernel_size=(32, 32)))
        model_ft = densenet
        input_img_size = 1024
    else:
        model_ft = VGGFeature(depth=depth, pooling=True, pretrained=True)
        input_img_size = 224
    model_ft.eval()
    model_ft = model_ft.to(device)

    dataset = Patches(slide, patch_coors, input_img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    fts = []
    with torch.no_grad():
        for _patches in dataloader:
            _patches = torch.squeeze(_patches, 1)
            _patches = _patches.to(device, non_blocking=True)
            _fts = model_ft(_patches)
            fts.append(_fts)
    fts = torch.cat(fts, dim=0)
    assert fts.size(0) == len(patch_coors)
    return fts


class Patches(Dataset):

    def __init__(self, slide: openslide, patch_coors, input_img_size=224) -> None:
        super().__init__()
        self.slide = slide
        self.patch_coors = patch_coors
        self.transform = transforms.Compose([
            transforms.Resize(input_img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx: int):
        coor = self.patch_coors[idx]
        img = self.slide.read_region((coor[0], coor[1]), 0, (coor[2], coor[3])).convert('RGB')
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.patch_coors)


def handle_slide(slide, num_sample=2000, patch_size=256, batch_size=256, cnn_base='resnet', cnn_depth=34, color_min=0.8):
    coordinates, bg_mask = sample_patch_coors(slide, num_sample=num_sample, patch_size=patch_size, color_min=color_min)
    features = extract_ft(slide, coordinates, depth=cnn_depth, batch_size=batch_size, cnn_base=cnn_base)
    return coordinates, features


def preprocess(svs_dir, result_dir):
    svs_list = get_files_type(svs_dir, 'svs', TMP)
    svs_list.sort()
    todo_list = check_todo(result_dir, svs_list, ['0.pkl', '0.npy', '1.pkl', '1.npy'])

    for svs_relative_path in tqdm(todo_list):
        svs_file = os.path.join(svs_dir, svs_relative_path)
        try:
            slide = openslide.open_slide(svs_file)

            coordinates, features = handle_slide(slide, num_sample=2000, patch_size=256)
            coordinates_file = get_save_path(result_dir, svs_relative_path, '0.pkl')
            with open(coordinates_file, 'wb') as fp:
                pickle.dump(coordinates, fp)
            np.save(get_save_path(result_dir, svs_relative_path, '0.npy'), features.cpu().numpy())

            coordinates, features = handle_slide(slide, num_sample=2000, patch_size=256)
            coordinates_file = get_save_path(result_dir, svs_relative_path, '1.pkl')
            with open(coordinates_file, 'wb') as fp:
                pickle.dump(coordinates, fp)
            np.save(get_save_path(result_dir, svs_relative_path, '1.npy'), features.cpu().numpy())

        except MemoryError as e:
            print('While handling ', svs_relative_path)
            print("find Memory Error, exit")
            exit()
        except Exception as e:
            print(e)
            print("failing in one image, continue")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess(SVS_DIR, RESULT_DIR)
