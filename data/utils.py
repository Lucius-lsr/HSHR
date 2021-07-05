# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 15:32
@Author  : Lucius
@FileName: utils.py
@Software: PyCharm
"""
import os


def get_files_type(directory, file_type):
    svs_list = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.' + file_type):
                relative_root = root[len(directory) + 1:]
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