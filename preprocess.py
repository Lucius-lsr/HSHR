# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/11 11:27
@Author  : Lucius
@FileName: preprocess.py
@Software: PyCharm
"""

import openslide
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.base_cnns import ResNetFeature, VGGFeature
import os
import pickle
from models.HyperG.utils.data.pathology import sample_patch_coors, draw_patches_on_slide

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


def get_files_type(directory, file_type):
    svs_list = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.' + file_type):
                relative_root = root[len(directory)+1:]
                svs_list.append(os.path.join(relative_root, file))
    return svs_list


def check_todo(svs_list, feature_list, coordinates_list):
    to_do_list = list()
    for svs_file in svs_list:
        feature_file = svs_file.replace('.svs', '.npy')
        coordinates_file = svs_file.replace('.svs', '.pkl')
        if feature_file not in feature_list or coordinates_file not in coordinates_list:
            to_do_list.append(svs_file)

    return to_do_list


def check_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return file_path


def preprocess(svs_dir, feature_dir, coordinate_dir, image_dir, batch_size=256, cnn_base='resnet', cnn_depth=34):
    svs_list = get_files_type(svs_dir, 'svs')
    feature_list = get_files_type(feature_dir, 'npy')
    coordinate_list = get_files_type(coordinate_dir, 'pkl')
    todo_list = check_todo(svs_list, feature_list, coordinate_list)

    for svs_relative_path in tqdm(todo_list):
        svs_file = os.path.join(svs_dir, svs_relative_path)

        coordinates, bg_mask = sample_patch_coors(svs_file, num_sample=2000, patch_size=256)
        image = draw_patches_on_slide(svs_file, coordinates, bg_mask)
        with open(check_dir(os.path.join(coordinate_dir, svs_relative_path.replace('.svs', '.pkl'))), 'wb') as fp:
            pickle.dump(coordinates, fp)
        image.save(check_dir(os.path.join(image_dir, svs_relative_path.replace('.svs', '.png'))))

        # features = extract_ft(svs_file, coordinates, depth=cnn_depth, batch_size=batch_size, cnn_base=cnn_base)
        # np.save(check_dir(os.path.join(feature_dir, svs_relative_path.replace('.svs', '.npy'))), features.cpu().numpy())


if __name__ == '__main__':
    SVS_DIR = '/lishengrui/TCGA/lusc_tcga'
    FEATURE_DIR = '/home/lishengrui/TCGA_experiment/lusc_tcga'
    COORDINATE_DIR = '/home/lishengrui/TCGA_experiment/lusc_tcga'
    IMAGE_DIR = '/home/lishengrui/TCGA_experiment/lusc_tcga'
    preprocess(SVS_DIR, FEATURE_DIR, COORDINATE_DIR, IMAGE_DIR)
