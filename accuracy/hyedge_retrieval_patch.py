"""
@Time    : 2022/1/17 14:03
@Author  : Lucius
@FileName: hyedge_retrieval_patch.py
@Software: PyCharm
"""
import torch
import numpy as np


def generate_incidence(key_list, top_idx, k):
    patch_size = len(key_list)
    incidence = np.zeros([patch_size, patch_size])
    logk = np.log(k+1)
    for q_id, top in enumerate(top_idx):
        for i in range(k):
            patch_id = top[i]
            incidence[q_id][patch_id] += (1 - (np.log(i+1) / logk))
    return incidence


def generate_sh(inc):
    return inc.dot(np.transpose(inc))


def generate_sv(inc):
    return np.transpose(inc).dot(inc)


def hyedge_similarity_patch(key_list, top_idx, patch_num=20, k=200):
    list_slide_id = []
    for idx, key in enumerate(key_list):
        if idx % patch_num == 0:
            slide_id = key.split('@')[-2]
            list_slide_id.append(slide_id)
        else:
            assert key_list[idx-1].split('@')[-2] == key.split('@')[-2]

    inc = generate_incidence(key_list, top_idx, k)
    sh = generate_sh(inc)
    sv = generate_sv(inc)
    simi = sh * sv
    dim = simi.shape[0]
    simi.reshape(int(dim / patch_num), patch_num, int(dim / patch_num), patch_num).mean(3).mean(1)
    _, slide_top_idx = torch.topk(torch.from_numpy(simi), simi.shape[0], dim=1, largest=True)

    return list_slide_id, slide_top_idx

