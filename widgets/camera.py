from imgui_bundle import imgui
import torch
import numpy as np

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.dict_utils import EasyDict
from splatviz_utils.cam_utils import (
    get_forward_vector,
    create_cam2world_matrix,
    get_origin,
    normalize_vecs,
)
from splatviz_utils.gui_utils.interface_imgui import Combo, Slider, InputTensor, InputFloat
from widgets.widget import Widget


class CamWidget(Widget):
    def __init__(self, viz, fov=12, radius=2.7, up_direction=1, device="cuda"):
        super().__init__(viz, "Camera")
        self.device = device

        # cam params
        self.cam_pos = torch.tensor([0.0, 0.0, -1.0], device=device)
        self.forward = torch.tensor([0.0, 0.0, 1.0], device=device)

        # controls
        self.last_drag_delta = imgui.ImVec2(0, 0)

        # momentum
        self.momentum_x = 0.0
        self.momentum_y = 0.0

        # cam control
        self.cam_mode_combo = Combo(viz, "camera_mode", ["Orbit", "WASD"])
        self.move_speed_slider = Slider(viz, "move_speed", 0.1, 0.001, 1, log=True)
        self.drag_speed_slider = Slider(viz, "drag_speed", 0.002, 0.001, 0.1, log=True)
        self.momentum_slider = Slider(viz, "momentum", 0.3, 0.0, 0.999)
        self.momentum_dropoff_slider = Slider(viz, "momentum_dropoff", 0.8, 0.0, 1.0)
        self.rotate_speed_slider = Slider(viz, "rotate_speed", 0.005, 0.002, 0.1, log=True)

        # cam matrix
        self.up_vector_tensor_input = InputTensor(viz, "up_vector", [0.0, up_direction, 0.0], device=device)
        self.fov_slider = Slider(viz, "fov", fov, 1, 180, format="%.2f °", add_to_args=True, with_input_field=True)
        self.yaw_input = InputFloat(viz, "yaw", np.pi)
        self.pitch_input = InputFloat(viz, "pitch", 0)
        self.radius_slider = Slider(viz, "radius", radius, 20, with_input_field=True)
        self.lookat_point_tensor = InputTensor(viz, "lookat_point", [0.0, 0.0, 0.0], device=device)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show: bool):
        viz = self.viz
        active_region = EasyDict(x=viz.pane_w, y=0, width=viz.content_width - viz.pane_w, height=viz.content_height)
        self.handle_dragging_in_window(**active_region)
        self.handle_mouse_wheel()
        self.handle_wasd()

        if show:
            imgui.text("Camera Controls")
            self.cam_mode_combo()
            if self.cam_mode_combo.value == "WASD":
                self.move_speed_slider()
            self.drag_speed_slider()
            self.momentum_slider()
            self.momentum_dropoff_slider()
            self.rotate_speed_slider()

            imgui.text("\nCamera Matrix")
            imgui.push_item_width(200)
            self.up_vector_tensor_input()
            imgui.same_line()
            if imgui_utils.button("Set current direction", width=viz.button_large_w):
                self.up_vector_tensor_input.value = -self.forward
                self.yaw_input.value = 0
                self.pitch_input.value = 0
            imgui.same_line()
            if imgui_utils.button("Flip", width=viz.button_w):
                self.up_vector_tensor_input.value *= -1
            self.fov_slider()

            if self.cam_mode_combo.value == "Orbit":
                self.yaw_input()
                self.pitch_input()

                self.radius_slider()
                imgui.same_line()
                if imgui_utils.button("Set to xyz stddev", width=viz.button_large_w) and "std_xyz" in viz.result.keys():
                    self.radius_slider.value = viz.result.std_xyz.item()

                self.lookat_point_tensor()
                imgui.same_line()
                if imgui_utils.button("Set to xyz mean", width=viz.button_large_w) and "mean_xyz" in viz.result.keys():
                    self.lookat_point_tensor.value = viz.result.mean_xyz
            imgui.pop_item_width()


        self.cam_params = create_cam2world_matrix(self.forward, self.cam_pos, self.up_vector_tensor_input.value)[0]
        viz.args.cam_params = self.cam_params

        if show:
            imgui.text("\nExtrinsics Matrix")
            imgui.input_float4("##extr0", self.cam_params.cpu().numpy().tolist()[0])
            imgui.input_float4("##extr1", self.cam_params.cpu().numpy().tolist()[1])
            imgui.input_float4("##extr2", self.cam_params.cpu().numpy().tolist()[2])
            imgui.input_float4("##extr3", self.cam_params.cpu().numpy().tolist()[3])

    def handle_dragging_in_window(self, x, y, width, height):
        if imgui.is_mouse_dragging(0):  # left mouse button
            new_delta = imgui.get_mouse_drag_delta(0)
            if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                delta = new_delta - self.last_drag_delta
                self.last_drag_delta = new_delta
                self.momentum_x = delta.x * self.rotate_speed_slider.value * (1 - self.momentum_slider.value) + (self.momentum_x * self.momentum_slider.value)
                self.momentum_y = delta.y * self.rotate_speed_slider.value * (1 - self.momentum_slider.value) + (self.momentum_y * self.momentum_slider.value)

        elif imgui.is_mouse_dragging(2) or imgui.is_mouse_dragging(1):  # right mouse button or middle mouse button
            new_delta = imgui.get_mouse_drag_delta(2)
            if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                delta = new_delta - self.last_drag_delta
                self.last_drag_delta = new_delta

                right = torch.linalg.cross(self.forward, self.up_vector_tensor_input.value)
                right = right / torch.linalg.norm(right)
                cam_up = torch.linalg.cross(right, self.forward)
                cam_up = cam_up / torch.linalg.norm(cam_up)

                x_change = right * -delta.x * self.drag_speed_slider.value
                y_change = cam_up * delta.y * self.drag_speed_slider.value
                self.cam_pos += x_change
                self.cam_pos += y_change
                if self.cam_mode_combo.value == "Orbit":
                    self.lookat_point_tensor.value += x_change
                    self.lookat_point_tensor.value += y_change
        else:
            self.last_drag_delta = imgui.ImVec2(0, 0)

        self.yaw_input.value += self.momentum_x
        self.pitch_input.value += self.momentum_y
        self.momentum_x *= self.momentum_dropoff_slider.value
        self.momentum_y *= self.momentum_dropoff_slider.value
        self.pitch_input.value = np.clip(self.pitch_input.value, -np.pi / 2, np.pi / 2)

    def handle_wasd(self):
        if self.cam_mode_combo.value == "WASD":
            self.forward = get_forward_vector(
                lookat_position=self.cam_pos,
                horizontal_mean=self.yaw_input.value + np.pi / 2,
                vertical_mean=self.pitch_input.value + np.pi / 2,
                radius=0.01,
                up_vector=self.up_vector_tensor_input.value,
            )
            self.sideways = torch.linalg.cross(self.forward, self.up_vector_tensor_input.value)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.cam_pos += self.forward * self.move_speed_slider.value
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.cam_pos -= self.sideways * self.move_speed_slider.value
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.cam_pos -= self.forward * self.move_speed_slider.value
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.cam_pos += self.sideways * self.move_speed_slider.value
            if "q" in self.viz.current_pressed_keys:
                self.cam_pos += self.up_vector_tensor_input.value * self.move_speed_slider.value
            if "e" in self.viz.current_pressed_keys:
                self.cam_pos -= self.up_vector_tensor_input.value * self.move_speed_slider.value

        elif self.cam_mode_combo.value == "Orbit":
            self.cam_pos = get_origin(
                self.yaw_input.value + np.pi / 2,
                self.pitch_input.value + np.pi / 2,
                self.radius_slider.value,
                self.lookat_point_tensor.value,
                up_vector=self.up_vector_tensor_input.value,
            )
            self.forward = normalize_vecs(self.lookat_point_tensor.value - self.cam_pos)

    def handle_mouse_wheel(self):
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= self.viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            if self.cam_mode_combo.value == "WASD":
                self.cam_pos += self.forward * self.move_speed_slider.value * wheel
            elif self.cam_mode_combo.value == "Orbit":
                self.radius_slider.value -= wheel / 10
