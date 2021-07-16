# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/11 11:27
@Author  : Lucius
@FileName: preprocess.py
@Software: PyCharm
"""
import sys
import openslide
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data.utils import *
from models.base_cnns import ResNetFeature, VGGFeature
import os
import pickle
from models.HyperG.utils.data.pathology import sample_patch_coors, draw_patches_on_slide, raw_img
import numpy as np
from time import sleep

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_ft(slide_dir: str, patch_coors, depth=34, batch_size=16, cnn_base='resnet'):
    slide = openslide.open_slide(slide_dir)

    if cnn_base == 'resnet':
        model_ft = ResNetFeature(depth=depth, pooling=True, pretrained=True)
    else:
        model_ft = VGGFeature(depth=depth, pooling=True, pretrained=True)
    model_ft = model_ft.to(device)
    model_ft.eval()

    dataset = Patches(slide, patch_coors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    fts = []
    for _patches in dataloader:
        _patches = _patches.to(device)
        with torch.no_grad():
            _fts = model_ft(_patches)
        fts.append(_fts)

    fts = torch.cat(fts, dim=0)
    assert fts.size(0) == len(patch_coors)
    return fts


class Patches(Dataset):

    def __init__(self, slide: openslide, patch_coors) -> None:
        super().__init__()
        self.slide = slide
        self.patch_coors = patch_coors
        self.transform = transforms.Compose([
            transforms.Resize(224),
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


def preprocess(svs_dir, feature_dir, coordinate_dir, image_dir, range_from, range_to, handle_type, batch_size=256,
               cnn_base='resnet',
               cnn_depth=34):
    svs_list = get_files_type(svs_dir, 'svs')
    svs_list.sort()
    all_size = len(svs_list)
    range_from = int(range_from * all_size)
    range_to = int(range_to * all_size)
    svs_list = svs_list[range_from: range_to]
    print('To preprocess in {} with total {}, from {} to {}'.format(svs_dir, len(svs_list), range_from, range_to))

    todo_list = None
    if handle_type == 'coordinate':
        todo_list = check_todo(svs_dir, svs_list, ['0.pkl', '1.pkl'])
    elif handle_type == 'feature':
        todo_list = check_todo(svs_dir, svs_list, ['0.npy', '1.npy'])
    else:
        print('handle type is wrong')
        exit()
    print(handle_type)
    print('With {} / {} remaining'.format(len(todo_list), len(svs_list)))

    for svs_relative_path in tqdm(todo_list):
        svs_file = os.path.join(svs_dir, svs_relative_path)
        try:
            for i in range(2):
                if handle_type == 'coordinate':
                    coordinates, bg_mask = sample_patch_coors(svs_file, num_sample=2000, patch_size=256)
                    syn_image, raw_image = draw_patches_on_slide(svs_file, coordinates, bg_mask)
                    with open(get_save_path(coordinate_dir, svs_relative_path, '{}.pkl'.format(i)), 'wb') as fp:
                        pickle.dump(coordinates, fp)
                    syn_image.save(get_save_path(image_dir, svs_relative_path, '{}.jpg'.format(i)), quality=0)
                    if i == 0:
                        raw_image.save(get_save_path(image_dir, svs_relative_path, 'raw.jpg'), quality=10)
                    raw_image.close()
                elif handle_type == 'feature':
                    pre_file = get_save_path(coordinate_dir, svs_relative_path, '{}.pkl'.format(i))
                    while not os.path.exists(pre_file):
                        print('waiting...')
                        sleep(10)
                    with open(get_save_path(coordinate_dir, svs_relative_path, '{}.pkl'.format(i)), 'rb') as fp:
                        coordinates = pickle.load(fp)
                    features = extract_ft(svs_file, coordinates, depth=cnn_depth, batch_size=batch_size,
                                          cnn_base=cnn_base)
                    np.save(get_save_path(feature_dir, svs_relative_path, '{}.npy'.format(i)), features.cpu().numpy())

        except Exception as e:
            print(e)
            print("failing in one image, continue")


# def tmp_supply_raw_image(svs_dir, image_dir):
#     svs_list = get_files_type(svs_dir, 'svs')
#     print(svs_list)
#     for svs_relative_path in tqdm(svs_list):
#         try:
#             svs_file = os.path.join(svs_dir, svs_relative_path)
#             raw_image = raw_img(svs_file)
#             raw_image.save(check_dir(os.path.join(image_dir, svs_relative_path.replace('.svs', '.jpg'))))
#             raw_image.close()
#         except Exception as e:
#             print(e)
#             print("failing in one image, continue")


if __name__ == '__main__':
    SVS_DIR = '/lishengrui/TCGA'
    FEATURE_DIR = '/home/lishengrui/TCGA_experiment/all_tcga'
    COORDINATE_DIR = '/home/lishengrui/TCGA_experiment/all_tcga'
    IMAGE_DIR = '/home/lishengrui/TCGA_experiment/all_tcga'

    argv_list = sys.argv
    assert len(argv_list) == 4, 'wrong arguments'
    assert 0 <= float(argv_list[1]) <= 1
    assert 0 <= float(argv_list[2]) <= 1
    assert argv_list[1] < argv_list[2]
    preprocess(SVS_DIR, FEATURE_DIR, COORDINATE_DIR, IMAGE_DIR, float(argv_list[1]), float(argv_list[2]), argv_list[3])
    # tmp_supply_raw_image(SVS_DIR, IMAGE_DIR)
