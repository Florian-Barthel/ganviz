import numpy as np
import torch

from splatviz_utils.cam_utils import get_default_intrinsics, get_default_extrinsics
from sklearn.decomposition import PCA


class LatentMapRandom:
    def __init__(self, cols_rows=10, device="cuda"):
        self.device = device
        self.cols_rows = cols_rows
        self.z_map = torch.randn([1, 512, cols_rows, cols_rows], device=device, dtype=torch.float)
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

    def load_w_map(self, mapping_network, truncation_psi):
        intrinsics = get_default_intrinsics().to(self.device)
        extrinsics = get_default_extrinsics().to(self.device)
        mapping_camera_params = torch.concat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        reshaped_z_map = self.z_map.permute(0, 2, 3, 1).reshape(-1, 512)
        mapping_camera_params = mapping_camera_params.repeat([reshaped_z_map.shape[0], 1])

        self.w_map = mapping_network(reshaped_z_map, mapping_camera_params, truncation_psi=truncation_psi)
        if len(self.w_map.shape) == 3:
            self.w_map = self.w_map[:, 0, :]
        self.w_map = self.w_map.reshape(1, self.cols_rows, self.cols_rows, -1).permute(0, 3, 1, 2)

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
