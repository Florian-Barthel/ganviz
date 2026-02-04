import copy
import pickle
import numpy as np
import torch
import torch.nn
from tqdm import tqdm

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render_simple
from gaussian_splatting.scene.cameras import CustomCam
from gan_inversion.inversion import Inversion
from gan_preprocessing.preprocess import Preprocessor
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict
from gan_helper.latent_vector import LatentMapRandom, LatentMapPCA
from gan_helper.view_conditioning import view_conditioning


class ValueTracker:
    def __init__(self):
        self.values = []

    def __call__(self, values):
        something_changed = False
        if len(self.values) != len(values):
            self.values = values
            return True

        for cache_value, new_value in zip(self.values, values):
            if type(cache_value) != type(new_value):
                something_changed = True
                break
            elif isinstance(new_value, dict):
                if not self.compare_dict(cache_value, new_value):
                    something_changed = True
                    break
            elif isinstance(new_value, torch.Tensor):
                if not torch.equal(cache_value, new_value):
                    something_changed = True
                    break
            elif isinstance(new_value, np.ndarray):
                if not np.array_equal(cache_value, new_value):
                    something_changed = True
                    break
            elif cache_value != new_value:
                something_changed = True
                break

        self.values = copy.deepcopy(values)
        return something_changed

    @staticmethod
    def compare_dict(dict1, dict2):
        for key in dict1.keys():
            if dict1[key] != dict2[key]:
                return False
        return True

class GANRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.generator = None
        self.latent_dim = 512 * 2
        self.last_latent = torch.zeros([1, self.latent_dim], device=self._device)
        self._current_pkl_file_path = ""
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        self.device = torch.device("cuda")
        self.last_truncation_psi = 1.0
        self.last_mapping_conditioning = "frontal"
        self.last_seed = 0

        self.inversion_generator = None
        self.inverter = Inversion()
        self.preprocess = Preprocessor()
        self.w_inversion = torch.randn([1, self.latent_dim], device=self._device)
        self.inversion_step = 0
        self.use_inversion_w = False
        self.ws = None

        self.last_latent_face = None
        self.last_latent_glasses = None
        self.last_latent_hair = None
        self.last_latent_rest = None
        self.last_latent_context = None

        self.tracker = ValueTracker()
        self.z_map = torch.randn([1, 512, 10, 10], device="cuda", dtype=torch.float)
        self.z_context_map = torch.randn([1, 8, 10, 10], device="cuda", dtype=torch.float)


    def set_latents(self, list_of_latents, pca_components=None, latent_space="Z"):
        latent_dict = {}
        for latent_pos in list_of_latents:
            if latent_pos.name == "z_context":
                latent = self.latent_maps[latent_pos.name].get_latent(latent_pos.x, latent_pos.y, latent_space="Z")
                latent_dict[latent_pos.name] = latent
            else:
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
        latent_space="Z",
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
        latent_bg={},
        latent_context={},

        hair_color=None,
        skin_color=None,
        cloth_color=None,
        bg_color=None,

        only_show="0",
        has_glasses=False,
        disable_hair=False,
        fix_ws=False,

        **other_args
    ):
        slider = EasyDict(slider)
        self.pca_latent = False
        self.load(ply_file_paths[0])

        rerender = self.tracker([
            has_glasses,
            latent_face,
            latent_glasses,
            latent_hair,
            latent_bg,
            latent_rest,
            latent_context,
            hair_color,
            skin_color,
            cloth_color,
            bg_color,
            only_show,
            seed,
            mapping_conditioning,
            truncation_psi,
            ply_file_paths[0]
        ])


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
        seed_changed = seed != self.last_seed
        self.last_mapping_conditioning = mapping_conditioning

        if seed_changed or self.latent_maps is None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.create_latent_maps()
            self.last_seed = seed

        # generator
        truncation_psi_changed = self.last_truncation_psi != truncation_psi
        if truncation_psi_changed and latent_space == "W":
            if not self.pca_latent:
                self.create_latent_maps(truncation_psi)
        self.last_truncation_psi = truncation_psi

        latent_dict = self.set_latents([latent_face, latent_glasses, latent_hair, latent_rest, latent_context], pca_components, latent_space)

        with torch.no_grad():
            if rerender or mapping_conditioning == "current" or run_inversion or run_tuning:
                gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)

                if self.use_inversion_w:
                    mapped_latent = self.w_inversion

                conditioning = torch.zeros(1, 27+30+30+30, device="cuda")
                conditioning[:, :25] = gan_camera_params
                conditioning[:, 25] = int(only_show)
                conditioning[:, 26] = int(has_glasses)
                conditioning[:, 27 : 27+30] = hair_color
                conditioning[:, 27+30 : 27+30+30] = skin_color
                conditioning[:, 27+30+30 : 27+30+30+30] = cloth_color

                components = ["ws_head", "ws_glasses", "ws_hair", "ws_rest"]
                components_index = [1, 2, 3, 4]
                if int(only_show) != 0:
                    components = [components[int(only_show)-1]]
                    components_index = [components_index[int(only_show)-1]]
                gaussian_params = {
                    "_xyz":         torch.empty((0, 3), device="cuda", dtype=torch.float32),
                    "_scaling":     torch.empty((0, 3), device="cuda", dtype=torch.float32),
                    "_rotation":    torch.empty((0, 4), device="cuda", dtype=torch.float32),
                    "_features_dc": torch.empty((0, 1, 3), device="cuda", dtype=torch.float32),
                    "_features_rest": torch.empty((0, 0, 3), device="cuda", dtype=torch.float32),
                    "_opacity":     torch.empty((0, 1), device="cuda", dtype=torch.float32),
                }
                for i, name in enumerate(components):
                    conditioning[:, 25] = components_index[i]
                    # ws_dict = {
                    #     "ws_combined": latent_dict[name][:, None, :].repeat(1, self.generator.combined_mapping.num_ws, 1),
                    #     "ws_context": latent_dict["ws_context"],
                    # }
                    # gan_result = self.generator.synthesis(ws_dict, c=conditioning, render_output=False, single_image=False) # fix single_image=True
                    ws_dict = {
                        "ws_combined": latent_dict[name][:, None, :].repeat(1, self.generator.combined_mapping.num_ws, 1),
                        "z_context": latent_dict["z_context"],
                    }
                    gan_result = self.generator.synthesis(ws_dict, c=conditioning, render_output=False, single_image=False) # fix single_image=True
                    cur_params = gan_result["gaussian_params"][0]
                    
                    gaussian_params["_xyz"] = torch.concat([gaussian_params["_xyz"], cur_params["_xyz"]], dim=0)
                    gaussian_params["_scaling"] = torch.concat([gaussian_params["_scaling"], cur_params["_scaling"]], dim=0)
                    gaussian_params["_rotation"] = torch.concat([gaussian_params["_rotation"], cur_params["_rotation"]], dim=0)
                    if name == "ws_glasses":
                        opacity = cur_params["_opacity"] - 20 * (1 - int(has_glasses))
                    else:
                        opacity = cur_params["_opacity"]
                    gaussian_params["_opacity"] = torch.concat([gaussian_params["_opacity"], opacity.float()], dim=0)
                    gaussian_params["_features_dc"] = torch.concat([gaussian_params["_features_dc"], cur_params["_features_dc"]], dim=0)
                    gaussian_params["_features_rest"] = torch.concat([gaussian_params["_features_rest"], cur_params["_features_rest"]], dim=0)

                    print()
                self.last_latent = latent_dict
                self.extract_gaussians(gaussian_params)

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

    def create_latent_maps(self, truncation_psi=1.0):
        latent_networks = {
            "ws_head":      (self.generator.combined_mapping, self.generator.template_mapping, 512, True),
            "ws_glasses":   (self.generator.combined_mapping, self.generator.template_mapping, 512, True),
            "ws_hair":      (self.generator.combined_mapping, self.generator.template_mapping, 512, True),
            "ws_rest":      (self.generator.combined_mapping, self.generator.template_mapping, 512, True),
            #"ws_context":   (self.generator.context_mapping,  512, False),
            "z_context":    (None, None, 8, False),
        }
        self.latent_maps = {}


        for key, (mapping_network, template_mapping, size, use_cond) in latent_networks.items():
            if self.pca_latent:
                self.latent_maps[key] = LatentMapPCA(size)
                self.latent_maps[key].load_w_map(mapping_network)
            else:
                if key == "z_context":
                    self.latent_maps[key] = LatentMapRandom(size, use_cond=use_cond)
                    self.latent_maps[key].load_z_map(z_map=self.z_context_map)
                else:
                    self.latent_maps[key] = LatentMapRandom(size, use_cond=use_cond)
                    if mapping_network.w_avg_beta is None:
                        self.latent_maps[key].load_w_map(mapping_network, template_mapping, z_map=self.z_map) # todo w
                    else:
                        self.latent_maps[key].load_w_map(mapping_network, template_mapping, z_map=self.z_map, truncation_psi=truncation_psi)

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
        gan_model = EasyDict(gan_result)#["gaussian_params"][0])
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
        self.generator.use_bg_gen = False
        self.generator.use_shape_context = False
        self.generator.use_light_context = False
        self.generator.context_dim = 0

        self._current_pkl_file_path = pkl_file_path
        self.create_latent_maps()
        # self.inverter.set_generator(self.generator)
        return True
