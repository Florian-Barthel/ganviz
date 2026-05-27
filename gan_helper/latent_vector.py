import numpy as np
import torch

from splatviz_utils.cam_utils import get_default_intrinsics, get_default_extrinsics
from sklearn.decomposition import PCA


class LatentMapRandom:
    def __init__(self, cols_rows=20, device="cuda"):
        self.device = device
        self.cols_rows = cols_rows
        self.z_map = torch.randn([1, 512, cols_rows, cols_rows], device=device, dtype=torch.float)
        self.w_map = None

    def get_latent(self, latent_x, latent_y, latent_space):
        latent_x = torch.tensor(latent_x, device=self.device, dtype=torch.float)
        latent_y = torch.tensor(latent_y, device=self.device, dtype=torch.float)
        if latent_space == "Z":
            z = self.sample_tiled(self.z_map, latent_x, latent_y)
            return z.reshape(1, 512)
        elif latent_space == "W":
            if self.w_map is None:
                raise AssertionError("call load_w_map(mapping_network) first)")
            w = self.sample_tiled(self.w_map, latent_x, latent_y)
            return w.reshape(1, 512)
        else:
            raise NotImplementedError

    def sample_tiled(self, latent_map, latent_x, latent_y):
        width = latent_map.shape[3]
        height = latent_map.shape[2]

        x = torch.remainder(latent_x + 1.0, 2.0) / 2.0 * width
        y = torch.remainder(latent_y + 1.0, 2.0) / 2.0 * height

        x0 = torch.floor(x).long() % width
        y0 = torch.floor(y).long() % height
        x1 = (x0 + 1) % width
        y1 = (y0 + 1) % height

        wx = (x - torch.floor(x)).reshape(1, 1)
        wy = (y - torch.floor(y)).reshape(1, 1)

        top_left = latent_map[:, :, y0, x0]
        top_right = latent_map[:, :, y0, x1]
        bottom_left = latent_map[:, :, y1, x0]
        bottom_right = latent_map[:, :, y1, x1]

        top = top_left * (1.0 - wx) + top_right * wx
        bottom = bottom_left * (1.0 - wx) + bottom_right * wx
        return top * (1.0 - wy) + bottom * wy

    def load_w_map(self, mapping_network, truncation_psi):
        intrinsics = get_default_intrinsics().to(self.device)
        extrinsics = get_default_extrinsics().to(self.device)
        mapping_camera_params = torch.concat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        reshaped_z_map = self.z_map.permute(0, 2, 3, 1).reshape(-1, 512)
        mapping_camera_params = mapping_camera_params.repeat([reshaped_z_map.shape[0], 1])

        self.w_map = mapping_network(reshaped_z_map, mapping_camera_params, truncation_psi=truncation_psi)[:, 0, :]
        self.w_map = self.w_map.reshape(1, self.cols_rows, self.cols_rows, 512).permute(0, 3, 1, 2)

class LatentMapPCA:
    def __init__(self, device="cuda"):
        self.device = device
        self.components = None
        self.means = None

    def get_latent(self, latent_x, latent_y, pca_components):
        if self.components is None:
            raise AssertionError("call LatentMapPCA.load_pca() first")
        resulting_latent = np.sum(self.components * pca_components[..., None], axis=0) + self.means
        return torch.tensor(resulting_latent, device=self.device, dtype=torch.float)[None, :]

    def load_w_map(self, mapping_network, num_samples=100_000):
        z = torch.randn([num_samples, 512], device=self.device, dtype=torch.float)
        intrinsics = get_default_intrinsics().to(self.device)
        extrinsics = get_default_extrinsics().to(self.device)
        mapping_camera = torch.concat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        mapping_camera = mapping_camera.repeat([num_samples, 1])
        w = mapping_network(z, mapping_camera).detach().cpu().numpy()[:, 0, :]

        pca = PCA(n_components=50)
        pca.fit(w)
        self.components = pca.components_
        self.means = pca.mean_
