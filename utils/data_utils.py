# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/29 15:32
@Author  : Lucius
@FileName: data_utils.py
@Software: PyCharm
"""
import os
import pickle


def get_files_type(directory, file_suffix, tmp):
    tmp_file = tmp + '/{}_{}.pkl'.format(directory.replace('/', '*'), file_suffix.replace('.', '*'))
    if os.path.exists(tmp_file):
        with open(tmp_file, 'rb') as f:
            svs_list = pickle.load(f)
    else:
        svs_list = list()
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_suffix):
                    relative_root = root[len(directory) + 1:]
                    svs_list.append(os.path.join(relative_root, file))
        with open(tmp_file, 'wb') as fp:
            pickle.dump(svs_list, fp)
    return svs_list


def check_todo(root, svs_list, to_dos):
    to_do_list = list()
    for svs_relative_path in svs_list:
        svs_dir = os.path.dirname(svs_relative_path)
        svs_name = os.path.basename(svs_relative_path)
        relative_dir = os.path.join(svs_dir, svs_name[:-4])
        result_dir = os.path.join(root, relative_dir)
        if not os.path.exists(result_dir):
            to_do_list.append(relative_dir)
        else:
            files = os.listdir(result_dir)
            for to_do in to_dos:
                if to_do not in files:
                    to_do_list.append(relative_dir)
                    break

    return to_do_list


def check_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return file_path


def get_save_path(root_dir, svs_relative_path, file_name):
    full_path = os.path.join(root_dir, svs_relative_path, file_name)
    file_dir = os.path.dirname(full_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return full_path


def patientId(path):
    return path.split("/")[-1].split('-')[2]