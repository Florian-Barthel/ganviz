from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

from imgui_bundle import imgui
import numpy as np
import torch
from PIL import Image
import sys
sys.path.append("gan_inversion")
sys.path.append("gan_preprocessing")
sys.path.append("gaussian_splatting")

torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)

from renderer.renderer_wrapper import RendererWrapper
from renderer.gan_renderer import GANRenderer
from splatviz_utils.gui_utils import imgui_window
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils import gl_utils
from splatviz_utils.gui_utils import text_utils
from splatviz_utils.gui_utils.constants import *
from splatviz_utils.dict_utils import EasyDict
from widgets import (
    edit,
    eval,
    performance,
    load_pkl,
    camera,
    save,
    latent,
    render,
)


class Splatviz(imgui_window.ImguiWindow):
    def __init__(self, gan_path):
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"

        # Widget interface.
        self.result = EasyDict()
        self.eval_result = ""
        self.eval_text = ""
        self.args = EasyDict()

        super().__init__(
            title="splatviz",
            window_width=1920,
            window_height=1080,
            font=self.regular_font_path,
            code_font=self.code_font_path,
            close_on_esc=False,
        )

        # Internals.
        self._last_error_print = None

        self.widgets = []
        update_all_the_time = False

        self.widgets = [
            load_pkl.LoadWidget(self, file_ending=".pkl"),
            camera.CamWidget(self),
            performance.PerformanceWidget(self),
            save.CaptureWidget(self),
            render.RenderWidget(self),
            edit.EditWidget(self),
            eval.EvalWidget(self),
            latent.LatentWidget(self),
        ]
        self.gan_path = gan_path
        sys.path.append(gan_path)
        renderer = GANRenderer()

        self.renderer = RendererWrapper(renderer, update_all_the_time)
        self._tex_img = None
        self._tex_obj = None
        self._desc_tex_obj = None
        self._desc_svg_path = Path(__file__).with_name("desc.svg")
        self._desc_aspect_ratio = self._get_svg_aspect_ratio(self._desc_svg_path)
        self.renderer_fullscreen = False
        self._pending_renderer_fullscreen = None

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()
        self.preprocessed_images = []

    def close(self):
        for widget in self.widgets:
            widget.close()
        super().close()

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print(f"\n{error}\n")
            self._last_error_print = error

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def _set_sizes(self):
        self.pane_w = 0 if self.renderer_fullscreen else max(self.content_width - self.content_height, 500)
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100
        self.label_w_large = round(self.font_size * 5.5) + 150

    def set_renderer_fullscreen(self, fullscreen):
        self._pending_renderer_fullscreen = bool(fullscreen)

    @staticmethod
    def _get_svg_aspect_ratio(svg_path):
        _x, _y, width, height = ElementTree.parse(svg_path).getroot().attrib["viewBox"].split()
        return float(width) / float(height)

    def get_fullscreen_layout(self):
        desc_width = min(round(self.content_height * self._desc_aspect_ratio), self.content_width // 2)
        return self.content_width - desc_width, desc_width

    def get_fullscreen_render_size(self):
        width, _desc_width = self.get_fullscreen_layout()
        height = self.content_height
        smaller_side = min(width, height)
        if smaller_side <= 1024:
            return width, height

        scale = 1024 / smaller_side
        return round(width * scale), round(height * scale)

    def _get_desc_texture(self):
        if self._desc_tex_obj is None:
            import cairosvg

            png_bytes = cairosvg.svg2png(url=str(self._desc_svg_path), output_height=self.monitor_display_resolution[1])
            image = np.asarray(Image.open(BytesIO(png_bytes)).convert("RGB"))
            self._desc_tex_obj = gl_utils.Texture(image=image, bilinear=True, mipmap=True)
        return self._desc_tex_obj

    def draw_frame(self):
        if self._pending_renderer_fullscreen is not None:
            self.renderer_fullscreen = self._pending_renderer_fullscreen
            self._pending_renderer_fullscreen = None
            self.set_fullscreen(self.renderer_fullscreen)
            self.skip_frame()

        self.begin_frame()
        self._set_sizes()
        if "f" in self.current_key_presses:
            self.set_renderer_fullscreen(not self.renderer_fullscreen)
        if self.renderer_fullscreen and "escape" in self.current_pressed_keys:
            self.set_renderer_fullscreen(False)
            self.current_pressed_keys.discard("escape")

        if self.renderer_fullscreen:
            for widget in self.widgets:
                widget(False)
            self.args.render_width, self.args.render_height = self.get_fullscreen_render_size()
        else:
            # Control pane
            imgui.set_next_window_pos(imgui.ImVec2(0, 0))
            imgui.set_next_window_size(imgui.ImVec2(self.pane_w, self.content_height))
            control_pane_flags = WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE
            imgui.begin("##control_pane", p_open=True, flags=control_pane_flags)

            # Widgets
            for widget in self.widgets:
                expanded, _visible = imgui_utils.collapsing_header(widget.name, default=widget.name == "Load")
                imgui.indent()
                widget(expanded)
                imgui.unindent()
            self.args.render_width = None
            self.args.render_height = None

            # imgui.show_style_editor()

        # Render
        if self.is_skipping_frames():
            pass
        else:
            self.renderer.set_args(**self.args)
            result = self.renderer.result
            if result is not None:
                self.result = result

        # Display
        max_w = self.content_width - self.pane_w
        if self.renderer_fullscreen:
            max_w, desc_w = self.get_fullscreen_layout()
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if "image" in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            tex = text_utils.get_texture(
                self.result.message,
                size=self.font_size,
                max_width=max_w,
                max_height=max_h,
                outline=2,
            )
            tex.draw(pos=pos, align=0.5, rint=True, color=1)

        if self.renderer_fullscreen:
            desc_tex = self._get_desc_texture()
            desc_zoom = min(desc_w / desc_tex.width, max_h / desc_tex.height)
            desc_pos = np.array([max_w + desc_w / 2, max_h / 2])
            desc_tex.draw(pos=desc_pos, zoom=desc_zoom, align=0.5, rint=True)

        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None

        if "preprocessed_images" in self.result:
            self.preprocessed_images = self.result.preprocessed_images

        # End frame.
        self._adjust_font_size()
        if not self.renderer_fullscreen:
            imgui.end()
        self.end_frame()
