from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.interface_imgui import LatentSpace, Slider, Combo, InputInt, CheckboxInput
from widgets.widget import Widget


class LatentWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Latent")
        self.latent_space_obj_face =    LatentSpace(viz,  name="latent_face", latent_name="ws_head", add_to_args=True)
        self.latent_space_obj_glasses = LatentSpace(viz, name="latent_glasses", latent_name="ws_glasses", add_to_args=True)
        self.latent_space_obj_hair =    LatentSpace(viz, name="latent_hair", latent_name="ws_hair", add_to_args=True)
        self.latent_space_obj_rest =    LatentSpace(viz, name="latent_rest", latent_name="ws_rest", add_to_args=True)
        self.latent_space_obj_shape =    LatentSpace(viz, name="latent_shape", latent_name="ws_shape", add_to_args=True)


        self.truncation_slider =        Slider(viz, "truncation_psi", value=1.0, min_val=0, max_val=1.0, add_to_args=True)
        self.cam_conditioning_combo =   Combo(viz, "mapping_conditioning", ["frontal", "zero", "current"], add_to_args=True)
        self.latent_space_combo =       Combo(viz, "latent_space", ["W", "Z"], add_to_args=True)
        self.seed_input_int =           InputInt(viz, "seed", 0, add_to_args=True)
        self.has_glasses_checkbox =     CheckboxInput(viz, "has_glasses", value=False, add_to_args=True)
        self.only_show_combo =          Combo(viz, "only_show", ["0", "1", "2", "3", "4"], add_to_args=True)


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            self.has_glasses_checkbox()
            self.latent_space_obj_face()
            self.latent_space_obj_glasses()
            self.latent_space_obj_hair()
            self.latent_space_obj_rest()
            self.latent_space_obj_shape()
            self.only_show_combo()

            self.truncation_slider()
            self.cam_conditioning_combo()
            self.latent_space_combo()
            self.seed_input_int()
