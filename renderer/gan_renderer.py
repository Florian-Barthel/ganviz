import copy
import pickle
import numpy as np
import torch
import torch.nn
from tqdm import tqdm

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render_simple
from gaussian_splatting.scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict
from gan_helper.latent_vector import LatentMapRandom, LatentMapPCA
from gan_helper.view_conditioning import view_conditioning

class GANRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.generator = None
        self.last_latent = torch.zeros([1, 512], device=self._device)
        self._current_pkl_file_path = ""
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        self.latent_map = None
        self.device = torch.device("cuda")
        self.last_truncation_psi = 1.0
        self.last_mapping_conditioning = "frontal"
        self.last_seed = 0

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        cam_cond_params,
        current_ply_names,
        background_color,
        latent_space="Z",
        img_normalize=False,
        latent_x=0.0,
        latent_y=0.0,
        render_width=None,
        render_height=None,
        save_ply_path=None,
        truncation_psi=1.0,
        mapping_conditioning="frontal",
        save_ply_grid_path=None,
        seed=0,
        inversion_images=[],
        flame_params=None,
        run_inversion=False,
        run_tuning=False,
        inversion_hyperparams={},
        tuning_hyperparams={},
        slider={},
        pca_components=None,
        **other_args
    ):
        slider = EasyDict(slider)
        self.pca_latent = False
        model_changed = self.load(ply_file_paths[0])

        cam_params = cam_params.to(self.device)
        cam_cond_params = cam_cond_params.to(self.device)
        mapping_conditioning_changed = mapping_conditioning != self.last_mapping_conditioning
        seed_changed = seed != self.last_seed
        self.last_mapping_conditioning = mapping_conditioning
        if seed_changed or self.latent_map is None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.create_latent_map()
            self.last_seed = seed

        # generator
        truncation_psi_changed = self.last_truncation_psi != truncation_psi
        if truncation_psi_changed and latent_space == "W":
            if not self.pca_latent:
                self.latent_map.load_w_map(self.generator.mapping, truncation_psi)

        if self.pca_latent:
            latent = self.latent_map.get_latent(latent_x, latent_y, pca_components)
        else:
            latent = self.latent_map.get_latent(latent_x, latent_y, latent_space=latent_space)
        latent_changed = not torch.equal(self.last_latent, latent)

        with torch.no_grad():
            if True or seed_changed or latent_changed or model_changed or truncation_psi_changed or mapping_conditioning_changed or mapping_conditioning == "current" or run_inversion or run_tuning:
                if latent_space == "Z":
                    cam_cond_params = view_conditioning(cam_cond_params, fov, mapping_conditioning)
                    mapped_latent = self.generator.mapping(latent, cam_cond_params, truncation_psi=truncation_psi)
                elif latent_space == "W":
                    mapped_latent = latent[:, None, :] .repeat(1, self.generator.mapping_network.num_ws, 1)

                gan_result = self.generator.synthesis(mapped_latent, cam_params, render_output=False)
                self.last_latent = latent
                self.extract_gaussians(gan_result)

        # edit 3DGS scene
        gs = copy.deepcopy(self.gaussian_model)
        exec(edit_text)

        # render 3DGS scene
        render_width = int(render_width or resolution)
        render_height = int(render_height or resolution)
        fov_rad = fov / 360 * 2 * np.pi
        fovx_rad = 2 * np.arctan(np.tan(fov_rad / 2) * render_width / max(render_height, 1))
        render_cam = CustomCam(render_width, render_height, fovy=fov_rad, fovx=fovx_rad, extr=cam_params)
        img = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to(self.device))["render"]

        # return / eval / save scene
        self._return_image(img, res, normalize=img_normalize)
        if save_ply_path is not None:
            self.save_ply(gs, save_ply_path)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

        if save_ply_grid_path is not None:
            self.save_ply_grid(cam_params, fov, latent_space, mapped_latent, mapping_conditioning, truncation_psi)

    def create_latent_map(self):
        if self.pca_latent:
            self.latent_map = LatentMapPCA()
            self.latent_map.load_w_map(self.generator.mapping)
        else:
            self.latent_map = LatentMapRandom()
            self.latent_map.load_w_map(self.generator.mapping, self.last_truncation_psi)

    def save_ply_grid(self, cam_params, fov, latent_space, mapped_latent, mapping_conditioning, truncation_psi, steps=16):
        xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, steps), np.linspace(-0.5, 0.5, steps))
        for i in tqdm(range(steps)):
            for j in range(steps):
                x = xs[i, j]
                y = ys[i, j]
                latent = self.latent_map.get_latent(x, y, latent_space=latent_space)
                gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)
                if latent_space == "Z":
                    mapped_latent = self.generator.mapping(latent, mapping_camera_params, truncation_psi=truncation_psi)
                elif latent_space == "W":
                    mapped_latent = latent[:, None, :].repeat(1, self.generator.mapping_network.num_ws, 1)
                gan_result = self.generator.synthesis(mapped_latent, gan_camera_params)
                self.last_latent = latent
                self.extract_gaussians(gan_result)
                self.save_ply(self.gaussian_model, f"./_ply_grid/model_c{i:02d}_r{j:02d}.ply")

    def extract_gaussians(self, gan_result):

        # gan_model = EasyDict(gan_result["gaussian_params"])
        gan_model = EasyDict(gan_result["gaussian_params"][0])

        self.gaussian_model._xyz = gan_model._xyz
        self.gaussian_model._features_dc = gan_model._features_dc
        self.gaussian_model._features_rest = gan_model._features_dc[:, 0:0]
        self.gaussian_model._scaling = gan_model._scaling
        self.gaussian_model._rotation = gan_model._rotation
        self.gaussian_model._opacity = gan_model._opacity

    def load(self, pkl_file_path):
        if pkl_file_path == self._current_pkl_file_path:
            return False
        if not pkl_file_path.endswith(".pkl"):
            return False

        with open(pkl_file_path, "rb") as input_file:
            save_file = pickle.load(input_file)
        self.generator = copy.deepcopy(save_file["G_ema"]).eval().requires_grad_(True).to(self.device)
        self._current_pkl_file_path = pkl_file_path
        self.create_latent_map()
        return True
