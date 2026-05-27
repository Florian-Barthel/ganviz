from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.interface_imgui import CheckboxInput, LatentSpace, Slider, Combo, InputInt
from widgets.widget import Widget


class LatentWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Latent")
        self.latent_space_obj = LatentSpace(viz, "latent", "_x", "_y", add_to_args=True)
        self.gamepad_latent_checkbox = CheckboxInput(viz, "xbox latent", True)
        self.gamepad_latent_speed_slider = Slider(viz, "xbox latent speed", value=0.5, min_val=0.05, max_val=5.0, log=True)
        self.gamepad_deadzone_slider = Slider(viz, "xbox deadzone", value=0.18, min_val=0.0, max_val=0.5)

        self.truncation_slider = Slider(viz, "truncation_psi", value=0.6, min_val=0, max_val=1.0, add_to_args=True)
        self.cam_conditioning_combo = Combo(viz, "mapping_conditioning", ["current", "frontal", "zero"], add_to_args=True)
        self.latent_space_combo = Combo(viz, "latent_space", ["Z", "W"], selected=1, add_to_args=True)
        self.seed_input_int = InputInt(viz, "seed", 0, add_to_args=True)


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        self.handle_gamepad()
        if show:
            self.latent_space_obj()
            self.gamepad_latent_checkbox()
            self.gamepad_latent_speed_slider()
            self.gamepad_deadzone_slider()
            self.truncation_slider()
            self.cam_conditioning_combo()
            self.latent_space_combo()
            self.seed_input_int()
        else:
            self.sync_latent_args()

    def handle_gamepad(self):
        if not self.gamepad_latent_checkbox.value or not self.viz.gamepad_connected:
            return

        x = self.apply_gamepad_deadzone(self.viz.gamepad_axes.get("left_x", 0.0))
        y = self.apply_gamepad_deadzone(self.viz.gamepad_axes.get("left_y", 0.0))
        if x == 0.0 and y == 0.0:
            return

        delta = self.gamepad_latent_speed_slider.value * self.viz.frame_delta
        self.latent_space_obj.x += x * delta
        self.latent_space_obj.y -= y * delta
        self.latent_space_obj.wrap()
        self.sync_latent_args()

    def apply_gamepad_deadzone(self, value):
        value = float(value)
        deadzone = self.gamepad_deadzone_slider.value
        if abs(value) <= deadzone:
            return 0.0
        return (abs(value) - deadzone) / (1.0 - deadzone) * (1 if value >= 0 else -1)

    def sync_latent_args(self):
        if self.latent_space_obj.add_to_args:
            setattr(
                self.viz.args,
                self.latent_space_obj.name + self.latent_space_obj.name_x,
                self.latent_space_obj.x,
            )
            setattr(
                self.viz.args,
                self.latent_space_obj.name + self.latent_space_obj.name_y,
                self.latent_space_obj.y,
            )
