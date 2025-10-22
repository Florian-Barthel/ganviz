import numpy as np
import torch
from imgui_bundle._imgui_bundle import implot, imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label, slider



class CheckboxInput:
    def __init__(self, viz, name, value=False, add_to_args=False):
        self.viz = viz
        self.name = name
        self.value = value
        self.format = format
        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)

    def __call__(self):
        label(self.name, self.viz.label_w)
        changed, self.value = imgui.checkbox("##" + self.name + "_checkbox", self.value)
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)
        return changed


class InputInt:
    def __init__(
        self,
        viz,
        name,
        value=0,
        add_to_args=False,
    ):
        self.viz = viz
        self.name = name
        self.value = value
        self.format = format
        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)

    def __call__(self):
        label(self.name, self.viz.label_w)
        _changed, self.value = imgui.input_int("##" + self.name + "_input_int", self.value)
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)

class InputFloat:
    def __init__(
            self,
            viz,
            name,
            value,
            add_to_args=False,
            format="%.3f"
    ):
        self.viz = viz
        self.name = name
        self.value = value
        self.format = format
        if isinstance(value, list):
            self.num_entries = len(self.value)
            if self.num_entries == 1:
                self.imgui_func = imgui.input_float
            elif self.num_entries == 2:
                self.imgui_func = imgui.input_float2
            elif self.num_entries == 3:
                self.imgui_func = imgui.input_float3
            elif self.num_entries == 4:
                self.imgui_func = imgui.input_float4
            else:
                raise ValueError("only supports a maximum of 4 entries")
        else:
            self.imgui_func = imgui.input_float

        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)

    def __call__(self):
        label(self.name, self.viz.label_w)
        changed, self.value = self.imgui_func("##" + self.name + "_input_float", self.value, format=self.format)
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)


class InputTensor:
    def __init__(
        self,
        viz,
        name,
        value,
        add_to_args=False,
        device="cuda",
        format="%.3f"
    ):
        self.device = device
        self.viz = viz
        self.name = name
        self.value = torch.tensor(value, device=device)
        self.format = format
        self.num_entries = self.value.shape[0]
        if self.num_entries == 1:
            self.imgui_func = imgui.input_float
        elif self.num_entries == 2:
            self.imgui_func = imgui.input_float2
        elif self.num_entries == 3:
            self.imgui_func = imgui.input_float3
        elif self.num_entries == 4:
            self.imgui_func = imgui.input_float4
        else:
            raise ValueError("only supports a maximum of 4 entries")

        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)

    def __call__(self):
        label(self.name, self.viz.label_w)
        list_value = self.value.tolist()
        changed, list_value = self.imgui_func("##" + self.name + "_input_tensor", list_value, format=self.format)
        if changed:
            self.value = torch.tensor(list_value, device=self.device)
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)



class Slider:
    def __init__(
        self,
        viz,
        name,
        value=0,
        min_val=-1,
        max_val=1,
        log=False,
        add_to_args=False,
        with_input_field=False,
        format="%.3f"
    ):
        self.viz = viz
        self.name = name
        self.viz = viz
        self.name = name
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.log = log
        self.format = format
        self.with_input_field = with_input_field

        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)

    def __call__(self):
        label(self.name, self.viz.label_w)
        self.value = slider(self.value, "##" + self.name + "_slider", self.min_val, self.max_val, log=self.log, format=self.format)
        if self.with_input_field:
            imgui.same_line()
            _changed, self.value = imgui.input_float("##" + self.name + "_input_field", self.value)
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)


class Combo:
    def __init__(self, viz, name, selection, selected=0, add_to_args=False):
        self.name = name
        self.viz = viz
        self.selection = selection
        self.selected = selected
        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.selection[self.selected])

    @property
    def value(self):
        return self.selection[self.selected]

    @value.setter
    def value(self, value):
        self.selection[self.selected] = value

    def __call__(self):
        label(self.name, width=self.viz.label_w)
        _, self.selected = imgui.combo("##" + self.name + "_combo", self.selected, self.selection)
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.selection[self.selected])


class LatentSpace:
    def __init__(self, viz, name, name_x, name_y, size=10, color=(1, 1, 1), add_to_args=False):
        self.viz = viz
        self.name = name
        self.name_x = name_x
        self.name_y = name_y
        self.x = 0.0
        self.y = 0.0
        self.size = size
        self.color = color
        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name + self.name_x, self.x)
            setattr(self.viz.args, self.name + self.name_y, self.y)

    def __call__(self):
        _clicked, dragging, dx, dy = imgui_utils.drag_button(f"Drag {self.name}", width=self.viz.button_w)
        if dragging:
            self.x += dx * 0.0005
            self.y -= dy * 0.0005

        label("Latent")
        with imgui_utils.item_width(self.viz.font_size * 8):
            changed, (x_man, y_man) = imgui.input_float2("##" + self.name + "_drag_xy", v=[self.x, self.y])
            if changed:
                self.x = x_man
                self.y = y_man

        if implot.begin_plot(self.name, [self.viz.pane_w // 2, self.viz.pane_w // 2]):
            implot.setup_axes_limits(-1, 1, -1, 1, True)
            _changed, self.x, self.y, _, _, _ = implot.drag_point(0, self.x, self.y, imgui.ImVec4([*self.color, 1]), self.size, out_clicked=True)
            implot.end_plot()
        self.x = np.clip(self.x, -1, 1)
        self.y = np.clip(self.y, -1, 1)

        if self.add_to_args:
            setattr(self.viz.args, self.name + self.name_x, self.x)
            setattr(self.viz.args, self.name + self.name_y, self.y)



