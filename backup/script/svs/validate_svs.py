# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/30 14:27
@Author  : Lucius
@FileName: validate_svs.py
@Software: PyCharm
"""
import os
import pickle
import openslide
from tqdm import tqdm


def get_files_type(directory, file_suffix):
    svs_list = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_suffix):
                relative_root = root[len(directory) + 1:]
                svs_list.append(os.path.join(relative_root, file))
    return svs_list


if __name__ == '__main__':
    # svs_dir = '/home2/lishengrui/tcga_result'
    # svs_list = get_files_type(svs_dir, 'svs')
    # svs_list.sort()
    # mag20, mag40 = [], []
    # for svs in tqdm(svs_list):
    #     try:
    #         slide = openslide.open_slide(os.path.join(svs_dir, svs))
    #         p = slide.properties
    #         mag = p['aperio.AppMag']
    #         if mag == '20':
    #             mag20.append(svs)
    #         if mag == '40':
    #             mag40.append(svs)
    #     except Exception:
    #         continue
    # print(len(mag20))
    # print(len(mag40))
    tmp_file = 'mag20.pkl'
    # with open(tmp_file, 'wb') as fp:
    #     pickle.dump(mag20, fp)

    with open(tmp_file, 'rb') as f:
        svs_list = pickle.load(f)
    permit_key = {}
    for svs in svs_list:
        t, uuid, name = svs.split('/')[0], svs.split('/')[1], svs.split('/')[2]
        permit_key[uuid] = 1
        permit_key[name] = 2
        print(t, uuid, name)
    with open('permit_key', 'wb') as fp:
        pickle.dump(permit_key, fp)
