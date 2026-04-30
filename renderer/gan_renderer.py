import copy
import pickle
import numpy as np
import torch
import torch.nn
from PIL import Image
from tqdm import tqdm

from conditioning import load_cond, bin_size
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render_simple
from gaussian_splatting.scene.cameras import CustomCam
from gan_inversion.inversion import Inversion
from gan_preprocessing.preprocess import Preprocessor
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict
from gan_helper.latent_vector import LatentMapRandom, LatentMapPCA
from gan_helper.view_conditioning import view_conditioning
from comp_gan.training.disco_generator import CGSGenerator


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
        self.already_saved = False
        self.generator: CGSGenerator = None
        self.latent_dim = 512 * 2
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

        self.tracker = ValueTracker()
        self.z_map = torch.randn([1, 512, 10, 10], device="cuda", dtype=torch.float)
        self.z_shape_map = torch.randn([1, 512, 10, 10], device="cuda", dtype=torch.float)
        self.z_context_map = torch.randn([1, 8, 10, 10], device="cuda", dtype=torch.float)
        self.color_cond = load_cond(100)


    def set_latents(self, list_of_latents, pca_components=None, latent_space="W"):
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
                    latent = self.latent_maps[latent_pos.name].get_latent(latent_pos.x, latent_pos.y, latent_space=latent_space, components_multiplier=latent_pos.val)
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
        latent_bg={},
        latent_context={},
        shape_context={},

        hair_color=None,
        hair_color_weight=None,
        skin_color=None,
        skin_color_weight=None,
        cloth_color=None,
        cloth_color_weight=None,
        bg_color=None,
        bg_color_weight=None,

        only_show="0",
        has_glasses=False,
        disable_hair=False,
        fix_ws=False,
        color_index=-1,
        show_ema=False,
        cam_angles_mat=None,
        generate_angles=False,
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
            shape_context,
            hair_color,
            skin_color,
            cloth_color,
            hair_color_weight,
            skin_color_weight,
            cloth_color_weight,
            bg_color,
            only_show,
            seed,
            mapping_conditioning,
            truncation_psi,
            ply_file_paths[0],
            color_index,
            generate_angles
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

        latent_dict = self.set_latents([latent_face, latent_glasses, latent_hair, latent_rest, latent_context, shape_context], pca_components, latent_space)
        # self.generate_grid(cam_params, fov)
        gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)
        # self.save_with_diff_hair(cam_params, fov, background_color, mapping_conditioning, gan_camera_params)
        #self.save_with_diff_glasses(cam_params, fov, background_color, mapping_conditioning)

        with torch.no_grad():
            if rerender or mapping_conditioning == "current" or run_inversion or run_tuning:
                gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)

                if self.use_inversion_w:
                    mapped_latent = self.w_inversion

                conditioning = torch.zeros(1, 27+bin_size*3, device="cuda")
                conditioning[:, :25] = gan_camera_params
                conditioning[:, 25] = int(only_show)
                conditioning[:, 26] = int(has_glasses)
                if color_index < 0:
                    conditioning[:, 27 : 27+bin_size] =               hair_color
                    conditioning[:, 27+bin_size : 27+bin_size*2] =         skin_color
                    conditioning[:, 27+bin_size*2 : 27+bin_size*3] =   cloth_color
                else:
                    conditioning[:, 27:] = self.color_cond[color_index]
                    conditioning[:, 27: 27 + bin_size] =                      hair_color_weight * hair_color + (1 - hair_color_weight) * conditioning[:, 27: 27 + bin_size]
                    conditioning[:, 27 + bin_size: 27 + bin_size*2] =            skin_color_weight * skin_color + (1 - skin_color_weight) * conditioning[:, 27 + bin_size: 27 + bin_size*2]
                    conditioning[:, 27 + bin_size*2: 27 + bin_size*3] =  cloth_color_weight * cloth_color + (1 - cloth_color_weight) * conditioning[:, 27 + bin_size*2: 27 + bin_size*3]

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
                    ws_dict = {
                        "ws_combined": latent_dict[name][:, None, :].repeat(1, self.generator.combined_mapping.num_ws, 1),
                        "ws_context": latent_dict["ws_context"],
                        "ws_shape": latent_dict["ws_shape"],
                    }
                    gan_result = self.generator.synthesis(ws_dict, c=conditioning, render_output=False, single_image=True)
                    # ws_dict = {
                    #     "ws_combined": latent_dict[name][:, None, :].repeat(1, self.generator.combined_mapping.num_ws, 1),
                    #     "z_context": latent_dict["z_context"],
                    # }
                    # gan_result = self.generator.synthesis(ws_dict, c=conditioning, render_output=False, single_image=True)
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
                    if name == "ws_head" and show_ema:

                        gaussian_params["_xyz"] = self.generator.ema_geometry
                        gaussian_params["_opacity"] = self.generator.ema_geometry_opa

                self.extract_gaussians(gaussian_params)


        # edit 3DGS scene
        gs = copy.deepcopy(self.gaussian_model)
        exec(edit_text)

        if generate_angles:
            conditioning[:, 25] = 0
            self.save_different_angles(cam_angles_mat, fov, resolution, gs, background_color.to(self.device))

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

        # self.save_with_and_without_glasses(cam_params, fov, mapping_conditioning)
        if save_ply_grid_path is not None:
            self.save_ply_grid(cam_params, fov, latent_space, mapped_latent, mapping_conditioning, truncation_psi)

    def create_latent_maps(self, truncation_psi=1.0):
        # mapping network, input latent dimension, use conditioning
        shape_mapping = self.generator.mapping_network_shape
        latent_networks = {
            "ws_head":      (self.generator.combined_mapping, 512, True),
            "ws_glasses":   (self.generator.combined_mapping, 512, True),
            "ws_hair":      (self.generator.combined_mapping, 512, True),
            "ws_rest":      (self.generator.combined_mapping, 512, True),
            "ws_context":   (self.generator.context_mapping,  512, False),
            "ws_shape":     (self.generator.mapping_network_shape, 512, False),
            #"z_context":    (None, None, 4, False),
        }
        self.latent_maps = {}


        for key, (mapping_network, size, use_cond) in latent_networks.items():
            if self.pca_latent:
                self.latent_maps[key] = LatentMapPCA(size)
                self.latent_maps[key].load_w_map(mapping_network)
            else:
                if key == "z_context":
                    self.latent_maps[key] = LatentMapRandom(size, use_cond=use_cond, cond=self.color_cond)
                    self.latent_maps[key].load_z_map(z_map=self.z_context_map)
                else:
                    self.latent_maps[key] = LatentMapRandom(size, use_cond=use_cond, cond=self.color_cond)
                    if mapping_network.w_avg_beta is None:
                        self.latent_maps[key].load_w_map(mapping_network, shape_mapping, z_map=self.z_map, z_shape_map=self.z_shape_map)
                    else:
                        self.latent_maps[key].load_w_map(mapping_network, shape_mapping, z_map=self.z_map, z_shape_map=self.z_shape_map, truncation_psi=truncation_psi)

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
                self.extract_gaussians(gan_result)
                self.save_ply(self.gaussian_model, f"./_ply_grid/model_c{i:02d}_r{j:02d}.ply")


    def save_different_angles(self, cam_angles_mat, fov, resolution, gs, background_color):
        with torch.no_grad():
            fov_rad = fov / 360 * 2 * np.pi

            id_ = np.random.randint(10000)
            for i, cam in enumerate(cam_angles_mat):
                render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=torch.tensor(cam).cuda().reshape([4,4]))
                img = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to(self.device))["render"]
                img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                Image.fromarray(img).save(f"./_angles/{id_}_{i}.png")

    def save_with_diff_glasses(self, cam_params, fov, background_color, mapping_conditioning):
        if self.already_saved:
            return
        colors = load_cond(1000)
        with torch.no_grad():
            for i in range(1000):
                latent = torch.randn([1, 512 * 3], device="cuda")
                for has_glasses in [0, 1]:
                    gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)
                    conditioning = torch.zeros(1, 27 + bin_size*3, device="cuda")
                    conditioning[:, :25] = gan_camera_params
                    conditioning[:, 25] = 0
                    conditioning[:, 26] = int(has_glasses)
                    conditioning[:, 27:] = colors[i]
                    mapped_latent = self.generator.mapping(latent, conditioning)
                    gan_result = self.generator.synthesis(mapped_latent, conditioning, render_output=True, random_bg=False)
                    np_image = (gan_result["image"].cpu().numpy()[0].transpose(1, 2, 0) + 1) / 2 * 255
                    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
                    Image.fromarray(np_image).save(f"./_glasses/{i}_{has_glasses}.png")
        self.already_saved = True


    def save_with_diff_hair(self, cam_params, fov, background_color, mapping_conditioning, gan_camera_params):
        if self.already_saved:
            return
        colors = load_cond(1000)
        with torch.no_grad():
            for i in range(1000):
                conditioning = torch.zeros(1, 27 + bin_size*3, device="cuda")
                conditioning[:, :25] = gan_camera_params
                conditioning[:, 25] = 0
                conditioning[:, 26] = 0
                conditioning[:, 27:] = colors[i]
                latent = torch.randn([1, 512 * 3], device="cuda")
                mapped_latent = self.generator.mapping(latent, conditioning)
                latent = torch.randn([1, 512 * 3], device="cuda")
                mapped_latent_diff = self.generator.mapping(latent, conditioning)

                latent_dict = [mapped_latent, mapped_latent_diff]
                for hair in [0, 1]:
                    gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)

                    gan_result = self.render_component_wise(resolution=512, fov=fov, conditioning=conditioning, cam_params=cam_params, has_glasses=False, background_color=background_color, latent_dict=latent_dict[0], latent_dict_hair=latent_dict[hair])
                    np_image = gan_result.cpu().numpy().transpose(1, 2, 0) * 255
                    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
                    Image.fromarray(np_image).save(f"./_hair/{i}_{hair}.png")
        self.already_saved = True

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
        self.generator = copy.deepcopy(save_file["G_ema"]).eval().requires_grad_(False).to(self.device)

        self._current_pkl_file_path = pkl_file_path
        self.create_latent_maps()
        # self.inverter.set_generator(self.generator)
        return True

    def generate_grid(
            self,
            cam_params,
            fov,
            resolution=1024,
            latent_space="W",
            mapping_conditioning="frontal",
    ):
        """
        Generates a 4x4 grid where:
          - rows vary head latent (ws_head)
          - columns vary hair latent (ws_hair)
        """

        device = self.device
        grid_size = 4


        # Fixed coordinates for sampling
        coords = torch.linspace(-0.5, 0.5, grid_size)
        head_latents = [self.latent_maps["ws_head"].get_latent(x.item(), 0.0, latent_space) for x in coords]
        hair_latents = [self.latent_maps["ws_hair"].get_latent(0.0, y.item(), latent_space) for y in coords]

        # Fixed latents (shared across grid)
        ws_glasses = self.latent_maps["ws_glasses"].get_latent(0.0, 0.0, latent_space)
        ws_rest = self.latent_maps["ws_rest"].get_latent(0.0, 0.0, latent_space)
        ws_context = self.latent_maps["ws_context"].get_latent(0.0, 0.0, latent_space)
        ws_shape = self.latent_maps["ws_shape"].get_latent(0.0, 0.0, latent_space)

        images = []

        gan_camera_params, _ = view_conditioning(
            cam_params.to(device), fov, mapping_conditioning
        )

        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(
            resolution,
            resolution,
            fovy=fov_rad,
            fovx=fov_rad,
            extr=cam_params.to(device),
        )
        dataset_conditioning = load_cond(100)[5:6]
        conditioning = torch.zeros(1, 27+bin_size*3, device=device)
        conditioning[:, 27:] = dataset_conditioning
        with torch.no_grad():
            for i in range(grid_size):
                row_imgs = []
                for j in range(grid_size):
                    latent_dict = {
                        "ws_head": head_latents[i],
                        "ws_hair": hair_latents[j],
                        "ws_glasses": ws_glasses,
                        "ws_rest": ws_rest,
                        "ws_context": ws_context,
                        "ws_shape": ws_shape,
                    }

                    gaussian_params = {
                        "_xyz": torch.empty((0, 3), device=device),
                        "_scaling": torch.empty((0, 3), device=device),
                        "_rotation": torch.empty((0, 4), device=device),
                        "_features_dc": torch.empty((0, 1, 3), device=device),
                        "_features_rest": torch.empty((0, 0, 3), device=device),
                        "_opacity": torch.empty((0, 1), device=device),
                    }

                    components = ["ws_head", "ws_glasses", "ws_hair", "ws_rest"]
                    component_ids = [1, 2, 3, 4]

                    for name, cid in zip(components, component_ids):
                        conditioning[:, 25] = cid
                        ws_dict = {
                            "ws_combined": latent_dict[name][:, None, :].repeat(1, self.generator.combined_mapping.num_ws, 1),
                            "ws_context": ws_context,
                            "ws_shape": ws_shape,
                        }

                        out = self.generator.synthesis(
                            ws_dict,
                            c=conditioning,
                            render_output=False,
                            single_image=True,
                        )["gaussian_params"][0]

                        for k in gaussian_params:
                            gaussian_params[k] = torch.cat(
                                [gaussian_params[k], out[k]], dim=0
                            )

                    self.extract_gaussians(gaussian_params)
                    img = render_simple(
                        viewpoint_camera=render_cam,
                        pc=self.gaussian_model,
                        bg_color=torch.ones(3, device=device),
                    )["render"]

                    row_imgs.append(img)
                images.append(torch.stack(row_imgs, dim=0))

        # (4, 4, 3, H, W)
        self.save_grid_image_manual(torch.stack(images, dim=0), path="grid.png")

    @staticmethod
    def save_grid_image_manual(grid, path):
        _, _, C, H, W = grid.shape
        grid = grid.permute(2, 0, 3, 1, 4)
        grid = grid.reshape(C, 4 * H, 4 * W)
        grid = grid.clamp(0, 1)
        img = (grid * 255).byte()
        img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        Image.fromarray(img).save(path)


    def render_component_wise(self, resolution, fov, conditioning, cam_params, has_glasses, background_color, latent_dict, latent_dict_hair):
        with torch.no_grad():
            components = ["ws_head", "ws_glasses", "ws_hair", "ws_rest"]
            components_index = [1, 2, 3, 4]

            gaussian_params = {
                "_xyz": torch.empty((0, 3), device="cuda", dtype=torch.float32),
                "_scaling": torch.empty((0, 3), device="cuda", dtype=torch.float32),
                "_rotation": torch.empty((0, 4), device="cuda", dtype=torch.float32),
                "_features_dc": torch.empty((0, 1, 3), device="cuda", dtype=torch.float32),
                "_features_rest": torch.empty((0, 0, 3), device="cuda", dtype=torch.float32),
                "_opacity": torch.empty((0, 1), device="cuda", dtype=torch.float32),
            }
            for i, name in enumerate(components):
                conditioning[:, 25] = components_index[i]
                ws_dict = {
                    "ws_combined": latent_dict["ws_combined"],
                    "ws_context": latent_dict["ws_context"],
                    "ws_shape": latent_dict["ws_shape"],
                }
                if name == "ws_hair":
                    ws_dict["ws_combined"] = latent_dict_hair["ws_combined"]
                gan_result = self.generator.synthesis(ws_dict, c=conditioning, render_output=False, single_image=True)
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

            self.extract_gaussians(gaussian_params)

        # edit 3DGS scene
        gs = copy.deepcopy(self.gaussian_model)

        # render 3DGS scene
        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
        img = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to(self.device))["render"]
        return img