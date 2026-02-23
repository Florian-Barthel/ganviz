import numpy as np
import time

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.interface_imgui import LatentSpace, Slider, Combo, InputInt
from splatviz_utils.gui_utils.easy_imgui import checkbox, label
from widgets.widget import Widget


class LatentWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Latent")
        self.latent_space_obj = LatentSpace(viz, "latent", "_x", "_y", add_to_args=True)

        self.truncation_slider = Slider(viz, "truncation_psi", value=0.7, min_val=0, max_val=1.0, add_to_args=True)
        self.cam_conditioning_combo = Combo(viz, "mapping_conditioning", ["frontal", "zero", "current"], add_to_args=True)
        self.latent_space_combo = Combo(viz, "latent_space", ["W", "Z"], add_to_args=True)
        self.seed_input_int = InputInt(viz, "seed", 0, add_to_args=True)
        self.animate = False
        self.speed = 10

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            label("Animation")
            self.animate = checkbox(self.animate, "animate")
            if self.animate:
                self.latent_space_obj.x = np.sin(time.time() / self.speed) * 0.5
                self.latent_space_obj.y = np.cos(time.time() / self.speed) * 0.5
            self.latent_space_obj()
            self.truncation_slider()
            self.cam_conditioning_combo()
            self.latent_space_combo()
            self.seed_input_int()
