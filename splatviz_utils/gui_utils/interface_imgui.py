import numpy as np
import torch
from imgui_bundle import ImVec2
from imgui_bundle._imgui_bundle import implot, imgui

from conditioning import bin_size
from splatviz_utils.dict_utils import EasyDict
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
            self.imgui_func = None

        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)

    def __call__(self):
        label(self.name, self.viz.label_w)
        list_value = self.value.tolist()
        if not self.imgui_func is None:
            changed, list_value = self.imgui_func("##" + self.name + "_input_tensor", list_value, format=self.format)
        else:
            changed = False
            for index in range(len(list_value)):
                imgui.push_item_width(40)
                entry_changed, list_value[index] = imgui.input_float("##" + self.name + "_input_tensor_" + str(index), list_value[index], format=self.format)
                imgui.same_line()
                imgui.pop_item_width()
                if entry_changed:
                    changed = True

            imgui.new_line()
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
    def __init__(self, viz, name, latent_name, size=20, color=(1, 1, 1), limits=(-1, 1), add_to_args=False, num_components=4):
        self.viz = viz
        self.name = name
        self.latent_name = latent_name
        self.pos = EasyDict(x=0.0, y=0.0, val=[0]*num_components, name=self.latent_name)
        self.size = size
        self.color = color
        self.add_to_args = add_to_args
        self.limits = limits

        self.checkpoints = []

        self.num_components = num_components
        self.min_val = -3
        self.max_val = 3

        if self.add_to_args:
            setattr(self.viz.args, self.name, self.pos)

    def __call__(self):
        if imgui.button(f"Save Checkpoint {self.name}", size=ImVec2(self.viz.label_w_large, 0)):
            self.checkpoints.append((self.pos.x, self.pos.y))

        for i, checkpoint in enumerate(self.checkpoints):
            if imgui.button(f"Load Checkpoint {i} {self.name}", size=ImVec2(self.viz.label_w_large, 0)):
                self.pos.x, self.pos.y = checkpoint

        if implot.begin_plot(self.name, [self.viz.pane_w // 3, self.viz.pane_w // 3]):
            implot.setup_axes_limits(self.limits[0], self.limits[1], self.limits[0], self.limits[1], True)
            _changed, self.pos.x, self.pos.y, _, _, _ = implot.drag_point(0, self.pos.x, self.pos.y, imgui.ImVec4([*self.color, 1]), self.size, out_clicked=True)
            implot.end_plot()
        self.pos.x = np.clip(self.pos.x, self.limits[0], self.limits[1])
        self.pos.y = np.clip(self.pos.y, self.limits[0], self.limits[1])

        label(self.name, self.viz.label_w)
        if imgui.button(f"Reset {self.name}", size=ImVec2(self.viz.label_w_large, 0)):
            self.pos.val = [0]*self.num_components
        imgui.new_line()

        for i in range(self.num_components):
            label(f"PCA{i}", self.viz.label_w)
            imgui.push_item_width(300)
            self.pos.val[i] = slider(self.pos.val[i], "##" + self.name + str(i) + "_slider",  self.min_val, self.max_val)
            imgui.pop_item_width()

            imgui.same_line()
            imgui.push_item_width(100)
            _changed, self.pos.val[i] = imgui.input_float("##" + self.name + str(i) + "_input_field", self.pos.val[i])
            imgui.pop_item_width()

        if self.add_to_args:
            setattr(self.viz.args, self.name, self.pos)



class ColorHist:
    def __init__(self, viz, name, num_colors=3, add_to_args=False, device="cuda"):
        self.viz = viz
        self.name = name
        self.num_colors = num_colors
        self.colors = np.ones([self.num_colors, 3])
        self.weight = 0
        self.colors[0] *= 0.5
        self.colors[1] *= 0.0

        self.device = device
        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self._calc_hist())
            setattr(self.viz.args, self.name + "_weight", self.weight)

    def _calc_hist(self):
        r_hist, _bin_borders = np.histogram(self.colors[:, 0], bins=bin_size // 3, range=(0, 1.0))
        g_hist, _bin_borders = np.histogram(self.colors[:, 1], bins=bin_size // 3, range=(0, 1.0))
        b_hist, _bin_borders = np.histogram(self.colors[:, 2], bins=bin_size // 3, range=(0, 1.0))

        r_hist = r_hist.astype(float) / 3
        g_hist = g_hist.astype(float) / 3
        b_hist = b_hist.astype(float) / 3

        # Save result
        result = torch.tensor(np.concatenate([r_hist, g_hist, b_hist]), device=self.device)[None, ...]
        return result

    def __call__(self):
        label(self.name, width=self.viz.label_w)

        imgui.new_line()
        for i in range(self.num_colors):
            imgui.same_line()
            imgui.push_item_width(300)
            changed, self.colors[i] = imgui.color_picker3("##" + self.name + f"_color_{i}", self.colors[i].tolist())
            imgui.pop_item_width()

        label(f"{self.name} Weight", self.viz.label_w)
        imgui.push_item_width(200)
        self.weight = slider(self.weight, "##" + self.name + f"_color_weight_slider", 0, 1)
        imgui.pop_item_width()

        if self.add_to_args:
            setattr(self.viz.args, self.name, self._calc_hist())
            setattr(self.viz.args, self.name + "_weight", self.weight)


class ColorHistYCbCr:
    def __init__(self, viz, name, num_colors=3, add_to_args=False, device="cuda"):
        self.viz = viz
        self.name = name
        self.num_colors = num_colors
        self.colors = np.ones([self.num_colors, 3])
        self.colors[0] *= 0.5
        self.colors[1] *= 0.0

        self.device = device
        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self._calc_hist())

    def _calc_hist(self):

        R = self.colors[:, 0] * 255
        G = self.colors[:, 1] * 255
        B = self.colors[:, 2] * 255
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.167 * R - 0.3313 * G - 0.5 * B + 128
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128


        Y_hist, _bin_borders =  np.histogram(Y,  bins=bin_size // 3, range=(0, 255))
        Cb_hist, _bin_borders = np.histogram(Cb, bins=bin_size // 3, range=(0, 255))
        Cr_hist, _bin_borders = np.histogram(Cr, bins=bin_size // 3, range=(0, 255))

        Y_hist = Y_hist.astype(float) / self.num_colors
        Cb_hist = Cb_hist.astype(float) / self.num_colors
        Cr_hist = Cr_hist.astype(float) / self.num_colors

        # Save result
        result = torch.tensor(np.concatenate([Y_hist, Cb_hist, Cr_hist]), device=self.device)[None, ...]
        #print(result)
        return result

    def __call__(self):
        label(self.name, width=self.viz.label_w)
        imgui.push_item_width(200)
        # imgui.new_line()
        for i in range(self.num_colors):
            imgui.same_line()

            changed, self.colors[i] = imgui.color_picker3("##" + self.name + f"_color_{i}", self.colors[i].tolist())

        imgui.pop_item_width()
        if self.add_to_args:
            setattr(self.viz.args, self.name, self._calc_hist())


class PCALatentSpace:
    def __init__(self, viz, name, num_components=50, add_to_args=False):
        self.viz = viz
        self.name = name
        self.num_components = num_components
        self.value = np.zeros(num_components)
        self.min_val = -3
        self.max_val = 3

        self.add_to_args = add_to_args
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)


    def __call__(self):
        label(self.name, self.viz.label_w)
        if imgui.button(f"Reset {self.name}", size=ImVec2(self.viz.label_w_large, 0)):
            self.value = np.zeros(self.num_components)
        imgui.new_line()
        for i in range(self.num_components):
            self.value[i] = slider(self.value[i], "##" + self.name + str(i) + "_slider",  self.min_val, self.max_val)
            imgui.same_line()
            _changed, self.value[i] = imgui.input_float("##" + self.name + str(i) + "_input_field", self.value[i])
        if self.add_to_args:
            setattr(self.viz.args, self.name, self.value)
