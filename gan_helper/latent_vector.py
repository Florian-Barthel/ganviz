import json

import numpy as np
import torch

from splatviz_utils.cam_utils import get_default_intrinsics, get_default_extrinsics
from sklearn.decomposition import PCA


def load_cond(num):
    cond = torch.zeros([num, 30 + 30 + 30], device="cuda", dtype=torch.float)
    file = "./color_hist_10_ffhqc_rgb_bg_fix.json"
    attributes = ["hair", "skin", "cloth"]

    with open(file, "r") as f:
        annotations = json.load(f)
        print("loading annotation")
        keys = list(annotations["hair"].keys())[:num]

    for i, key in enumerate(keys):
        for offset, attribute in enumerate(attributes):
            if key in annotations[attribute].keys():
                cond[i, 30 * offset: 30 * (offset + 1)] = torch.tensor(annotations[attribute][key], device="cuda", dtype=torch.float)
    return cond


class LatentMapRandom:
    def __init__(self, latent_dim, use_cond=True, cols_rows=10, device="cuda"):
        self.device = device
        self.latent_dim = latent_dim
        self.cols_rows = cols_rows
        self.z_map = torch.randn([1, latent_dim, cols_rows, cols_rows], device=device, dtype=torch.float)
        self.use_cond = use_cond

        self.c_map = None
        if self.use_cond:
            self.c_map = load_cond(num=cols_rows * cols_rows)
        self.w_map = None

    def get_latent(self, latent_x, latent_y, latent_space):
        latent_x = torch.tensor(latent_x, device=self.device, dtype=torch.float)
        latent_y = torch.tensor(latent_y, device=self.device, dtype=torch.float)
        position = torch.stack([latent_x, latent_y]).reshape(1, 1, 1, 2)
        if latent_space == "Z":
            z = torch.nn.functional.grid_sample(self.z_map, position, padding_mode="reflection", align_corners=False)
            return z.reshape(1, -1)
        elif latent_space == "W":
            if self.w_map is None:
                raise AssertionError("call load_w_map(mapping_network) first)")
            w = torch.nn.functional.grid_sample(self.w_map, position, padding_mode="reflection", align_corners=False)
            return w.reshape(1, -1)
        else:
            raise NotImplementedError

    def load_z_map(self, z_map=None):
        if z_map is not None:
            self.z_map = z_map
        return self.z_map

    def load_w_map(self, mapping_network, z_map=None, truncation_psi=1.0):
        if z_map is not None:
            self.z_map = z_map
        reshaped_z_map = self.z_map.permute(0, 2, 3, 1).reshape(self.cols_rows * self.cols_rows, -1)
        cond = self.c_map if self.use_cond else None
        self.w_map = mapping_network(reshaped_z_map, c=cond, truncation_psi=truncation_psi)
        if len(self.w_map.shape) == 3:
            self.w_map = self.w_map[:, 0, :]
        self.w_map = self.w_map.reshape(1, self.cols_rows, self.cols_rows, -1).permute(0, 3, 1, 2)

class LatentMapPCA:
    def __init__(self, latent_dim, device="cuda"):
        self.device = device
        self.components = None
        self.means = None
        self.latent_dim = latent_dim

    def get_latent(self, latent_x, latent_y, pca_components):
        if self.components is None:
            raise AssertionError("call LatentMapPCA.load_pca() first")
        resulting_latent = np.sum(self.components * pca_components[..., None], axis=0) + self.means
        return torch.tensor(resulting_latent, device=self.device, dtype=torch.float)[None, :]

    def load_w_map(self, mapping_network, num_samples=100_000):
        z = torch.randn([num_samples, self.latent_dim], device=self.device, dtype=torch.float)
        intrinsics = get_default_intrinsics().to(self.device)
        extrinsics = get_default_extrinsics().to(self.device)
        mapping_camera = torch.concat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        mapping_camera = mapping_camera.repeat([num_samples, 1])
        w = mapping_network(z, mapping_camera).detach().cpu().numpy()[:, 0, :]

        pca = PCA(n_components=50)
        pca.fit(w)
        self.components = pca.components_
        self.means = pca.mean_
