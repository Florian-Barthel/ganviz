import json

import torch
from tqdm import tqdm

bin_size = 5 * 3

def load_cond(num):
    cond = torch.zeros([num, bin_size + bin_size + bin_size], device="cuda", dtype=torch.float)
    if bin_size == 5 * 3:
        file = "./color_hist_5_ffhqc_rgb.json"
    elif bin_size == 10 * 3:
        file = "./color_hist_10_ffhqc_rgb_bg_fix.json"
    attributes = ["hair", "skin", "cloth"]

    with open(file, "r") as f:
        annotations = json.load(f)
        print("loading annotation")
        keys = list(annotations["hair"].keys())[:num]

    for i, key in tqdm(enumerate(keys)):
        for offset, attribute in enumerate(attributes):
            if key in annotations[attribute].keys():
                cond[i, bin_size * offset: bin_size * (offset + 1)] = torch.tensor(annotations[attribute][key], device="cuda", dtype=torch.float)
    return cond