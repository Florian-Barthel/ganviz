from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.interface_imgui import LatentSpace, Slider, Combo, InputInt, CheckboxInput, ColorHist, PCALatentSpace
from widgets.widget import Widget


class LatentWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Latent")

        self.latent_space_obj_face =            LatentSpace(viz,  name="latent_face",   latent_name="ws_head",      add_to_args=True)
        self.latent_space_obj_glasses =         LatentSpace(viz, name="latent_glasses", latent_name="ws_glasses",   add_to_args=True)
        self.latent_space_obj_hair =            LatentSpace(viz, name="latent_hair",    latent_name="ws_hair",      add_to_args=True)
        self.latent_space_obj_rest =            LatentSpace(viz, name="latent_rest",    latent_name="ws_rest",      add_to_args=True)
        self.latent_space_obj_context =         LatentSpace(viz, name="latent_context", latent_name="ws_context",   add_to_args=True)
        self.latent_space_obj_shape_context =   LatentSpace(viz, name="shape_context",  latent_name="ws_shape",     add_to_args=True)

        self.truncation_slider =        Slider(viz, "truncation_psi", value=1.0, min_val=0, max_val=1.0, add_to_args=True)
        # self.cam_conditioning_combo =   Combo(viz, "mapping_conditioning", ["frontal", "zero", "current"], add_to_args=True)
        # self.latent_space_combo =       Combo(viz, "latent_space", ["W", "Z"], selected=0, add_to_args=True)
        self.seed_input_int =           InputInt(viz, "seed", 0, add_to_args=True)
        self.has_glasses_checkbox =     CheckboxInput(viz, "has_glasses", value=True, add_to_args=True)
        self.only_show_combo =          Combo(viz, "only_show", ["0", "1", "2", "3", "4"], add_to_args=True)


        self.hair_color_input =         ColorHist(viz, "hair_color",    add_to_args=True)
        self.skin_color_input =         ColorHist(viz, "skin_color",    add_to_args=True)
        self.cloth_color_input =        ColorHist(viz, "cloth_color",   add_to_args=True)
        self.color_index = InputInt(viz, "color_index", 1, add_to_args=True)

        self.show_ema_checkbox = CheckboxInput(viz, "show_ema", value=False, add_to_args=True)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            self.latent_space_obj_face()
            self.skin_color_input()

            self.has_glasses_checkbox()
            self.latent_space_obj_glasses()

            self.latent_space_obj_hair()
            self.hair_color_input()

            self.latent_space_obj_rest()
            self.cloth_color_input()

            self.latent_space_obj_context()
            self.latent_space_obj_shape_context()

            self.only_show_combo()

            self.truncation_slider()
            # self.cam_conditioning_combo()
            # self.latent_space_combo()
            self.seed_input_int()
            self.color_index()
            self.show_ema_checkbox()