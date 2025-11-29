from imgui_bundle import imgui
import numpy as np

from splatviz_utils.gui_utils import imgui_window, gl_utils


class StereoWindow(imgui_window.ImguiWindow):
    def __init__(self):
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"

        super().__init__(
            title="splatviz stereo",
            window_width=1024*2,
            window_height=1024,
            font=self.regular_font_path,
            code_font=self.code_font_path,
        )

        self.code_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self.regular_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        # self._imgui_renderer.refresh_font_texture()

        # Internals.
        self._last_error_print = None

        self._tex_img = None
        self._tex_obj = None

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()
        self.image = None

    def close(self):
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
        self.pane_w = max(self.content_width - self.content_height, 500)
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100
        self.label_w_large = round(self.font_size * 5.5) + 150

    def draw_frame(self):
        self.begin_frame()
        self._set_sizes()
        print("draw stereo")

        # Display
        max_w = 2048
        max_h = 1024
        pos = np.array([max_w / 2, max_h / 2])
        if self._tex_img is not self.image:
            self._tex_img = self.image
            if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
            else:
                self._tex_obj.update(self._tex_img)
        zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
        self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)


        # End frame.
        self._adjust_font_size()
        self.end_frame()