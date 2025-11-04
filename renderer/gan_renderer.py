import copy
import pickle
import numpy as np
import torch
import torch.nn
from tqdm import tqdm

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render_simple
from gaussian_splatting.scene.cameras import CustomCam
# from gan_inversion.inversion import Inversion
from gan_preprocessing.preprocess import Preprocessor
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict
from gan_helper.latent_vector import LatentMapRandom, LatentMapPCA
from gan_helper.view_conditioning import view_conditioning

example_hair_hist = torch.tensor(
    data=[0.08009077323296618, 0.21680411800520286, 0.23794763934244756, 0.2336303758233243, 0.16206343056401173, 0.03747163336469807, 0.012564343831294626, 0.013062489621962695, 0.006254497149499087, 0.00011069906459290419, 0.03857862401062711, 0.1543698455748049, 0.23208058891902364, 0.3103448275862069, 0.174461725798417, 0.028947805391044447, 0.019150938174572425, 0.0172137045441966, 0.019095588642275972, 0.005756351358831018, 0.05180716222947916, 0.19914761720263463, 0.274976476448774, 0.2685559307023856, 0.10510876183096253, 0.03149388387668124, 0.018376044722422095, 0.02009188022361211, 0.02086677367576244, 0.009575469087286213],
    device="cuda",
)

example_face_hist = torch.tensor(
    data=[0.0022061492920774106, 0.005641039962020721, 0.02418386439163339, 0.21427574073556926, 0.25261805691306655, 0.26942947303750453, 0.22751263648803374, 0.004077187299282303, 5.5851880812086345e-05, 0.0, 0.002848445921416404, 0.005194224915524031, 0.05093691530062275, 0.2760758468541428, 0.25954369013376527, 0.2692060655142562, 0.13270406880951716, 0.0032673350275070514, 0.00022340752324834538, 0.0, 0.004524002345778994, 0.06872573933927224, 0.23153397190650396, 0.17875394453908236, 0.2358345667290346, 0.22876930380630567, 0.04853528442570303, 0.0029880756234466196, 0.00033511128487251807, 0.0],
    device="cuda",
)

example_cloth_hist = torch.tensor(
    data=[0.0022061492920774106, 0.005641039962020721, 0.02418386439163339, 0.21427574073556926, 0.25261805691306655, 0.26942947303750453, 0.22751263648803374, 0.004077187299282303, 5.5851880812086345e-05, 0.0, 0.002848445921416404, 0.005194224915524031, 0.05093691530062275, 0.2760758468541428, 0.25954369013376527, 0.2692060655142562, 0.13270406880951716, 0.0032673350275070514, 0.00022340752324834538, 0.0, 0.004524002345778994, 0.06872573933927224, 0.23153397190650396, 0.17875394453908236, 0.2358345667290346, 0.22876930380630567, 0.04853528442570303, 0.0029880756234466196, 0.00033511128487251807, 0.0],
    device="cuda",
)

class GANRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.generator = None
        self.latent_dim = 512 * 2
        self.last_latent = torch.zeros([1, self.latent_dim], device=self._device)
        self._current_pkl_file_path = ""
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        self.latent_map = None
        self.device = torch.device("cuda")
        self.last_truncation_psi = 1.0
        self.last_mapping_conditioning = "frontal"
        self.last_seed = 0

        # self.inversion_generator = None
        # self.inverter = Inversion()
        # self.preprocess = Preprocessor()
        # self.w_inversion = torch.randn([1, self.latent_dim], device=self._device)
        self.inversion_step = 0
        self.use_inversion_w = False

    def set_latents(self, list_of_latents, pca_components=None, latent_space="W"):
        latent_dict = {}
        for latent_pos in list_of_latents:
            if self.pca_latent:
                latent = self.latent_maps[latent_pos.name].get_latent(latent_pos.x, latent_pos.y, pca_components)
                latent_dict[latent_pos.name] = latent
            else:
                latent = self.latent_maps[latent_pos.name].get_latent(latent_pos.x, latent_pos.y, latent_space=latent_space)
                latent_dict[latent_pos.name] = latent
        return latent_dict



    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        current_ply_names,
        background_color,
        latent_space="W",
        img_normalize=False,
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
        render_glasses=False,

        latent_face={},
        latent_glasses={},
        latent_hair={},
        latent_rest={},
        latent_shape={},

        only_show="0",
        has_glasses=False,

        **other_args
    ):
        slider = EasyDict(slider)
        self.pca_latent = False
        model_changed = self.load(ply_file_paths[0])

        if len(inversion_images) > 0:
            preprocessed = self.preprocess(inversion_images, target_size=self.generator.resolution)
            res.preprocessed_images = preprocessed["cropped_images"]
            self.inverter.set_targets(images=preprocessed["cropped_images"], cams=preprocessed["cams"])

        if run_inversion:
            self.w_inversion, loss = self.inverter.step_w(self.generator, inversion_hyperparams)
        if run_tuning:
            self.w_inversion, loss = self.inverter.step_pti(self.generator, tuning_hyperparams)

        self.use_inversion_w = run_inversion or run_tuning

        cam_params = cam_params.to(self.device)
        mapping_conditioning_changed = mapping_conditioning != self.last_mapping_conditioning
        seed_changed = seed != self.last_seed
        self.last_mapping_conditioning = mapping_conditioning
        if seed_changed or self.latent_map is None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.create_latent_maps()
            self.last_seed = seed

        # generator
        truncation_psi_changed = self.last_truncation_psi != truncation_psi
        if truncation_psi_changed and latent_space == "W":
            if not self.pca_latent:
                self.latent_map.load_w_map(self.generator.mapping, truncation_psi)

        latent_dict = self.set_latents([latent_face, latent_glasses, latent_hair, latent_rest, latent_shape], pca_components, latent_space)

        with torch.no_grad():
            if seed_changed or model_changed or truncation_psi_changed or mapping_conditioning_changed or mapping_conditioning == "current" or run_inversion or run_tuning:
                gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)
                if latent_space == "Z":
                    raise NotImplementedError
                    # mapped_latent = self.generator.mapping(latent, mapping_camera_params, truncation_psi=truncation_psi)
                elif latent_space == "W":
                    mapped_latent = latent_dict#[:, None, :] .repeat(1, self.generator.mapping_network.num_ws, 1)

                if self.use_inversion_w:
                    mapped_latent = self.w_inversion

                conditioning = torch.zeros(1, 87 + 30, device="cuda")
                conditioning[:, :25] = gan_camera_params
                conditioning[:, 25] = int(only_show)
                conditioning[:, 26] = int(has_glasses)
                conditioning[:, 27:   27+30] = example_hair_hist
                conditioning[:, 27+30:27+60] = example_face_hist
                conditioning[:, 27+60:27+90] = example_cloth_hist

                print("rendering")

                gan_result = self.generator.synthesis(mapped_latent, c=conditioning, render_output=False)
                self.last_latent = latent_dict
                self.extract_gaussians(gan_result)

        # edit 3DGS scene
        gs = copy.deepcopy(self.gaussian_model)
        exec(edit_text)

        # render 3DGS scene
        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
        img = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to(self.device))["render"]

        # return / eval / save scene
        self._return_image(img, res, normalize=img_normalize)
        if save_ply_path is not None:
            self.save_ply(gs, save_ply_path)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

        if save_ply_grid_path is not None:
            self.save_ply_grid(cam_params, fov, latent_space, mapped_latent, mapping_conditioning, truncation_psi)

    def create_latent_maps(self):
        self.latent_networks = {
            "ws_head": self.generator.mapping_network,
            "ws_glasses": self.generator.mapping_network_glasses,
            "ws_hair": self.generator.mapping_network_hair,
            "ws_rest": self.generator.mapping_network_rest,
            "ws_shape": self.generator.mapping_network_shape,
        }
        self.latent_maps = {}

        for key, mapping_network in self.latent_networks.items():
            if self.pca_latent:
                self.latent_maps[key] = LatentMapPCA(512)
                self.latent_maps[key].load_w_map(mapping_network)
            else:
                self.latent_maps[key] = LatentMapRandom(512)
                self.latent_maps[key].load_w_map(mapping_network, self.last_truncation_psi)

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
        del save_file
        self._current_pkl_file_path = pkl_file_path
        self.create_latent_maps()
        # self.inverter.set_generator(self.generator)
        return True
