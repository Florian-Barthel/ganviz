import cv2
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.interface_imgui import InputInt, InputTensor, CheckboxInput, Combo
from widgets.widget import Widget

colormaps = [
    ("NONE", None),
    ("COLORMAP_AUTUMN", cv2.COLORMAP_AUTUMN),
    ("COLORMAP_BONE", cv2.COLORMAP_BONE),
    ("COLORMAP_JET", cv2.COLORMAP_JET),
    ("COLORMAP_WINTER", cv2.COLORMAP_WINTER),
    ("COLORMAP_RAINBOW", cv2.COLORMAP_RAINBOW),
    ("COLORMAP_OCEAN", cv2.COLORMAP_OCEAN),
    ("COLORMAP_SUMMER", cv2.COLORMAP_SUMMER),
    ("COLORMAP_SPRING", cv2.COLORMAP_SPRING),
    ("COLORMAP_COOL", cv2.COLORMAP_COOL),
    ("COLORMAP_HSV", cv2.COLORMAP_HSV),
    ("COLORMAP_PINK", cv2.COLORMAP_PINK),
    ("COLORMAP_HOT", cv2.COLORMAP_HOT),
    ("COLORMAP_PARULA", cv2.COLORMAP_PARULA),
    ("COLORMAP_MAGMA", cv2.COLORMAP_MAGMA),
    ("COLORMAP_INFERNO", cv2.COLORMAP_INFERNO),
    ("COLORMAP_PLASMA", cv2.COLORMAP_PLASMA),
    ("COLORMAP_VIRIDIS", cv2.COLORMAP_VIRIDIS),
    ("COLORMAP_CIVIDIS", cv2.COLORMAP_CIVIDIS),
    ("COLORMAP_TWILIGHT", cv2.COLORMAP_TWILIGHT),
    ("COLORMAP_TWILIGHT_SHIFTED", cv2.COLORMAP_TWILIGHT_SHIFTED),
    ("COLORMAP_TURBO", cv2.COLORMAP_TURBO),
    ("COLORMAP_DEEPGREEN", cv2.COLORMAP_DEEPGREEN),
]

class RenderWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Render")
        self.render_alpha = False
        self.render_depth = False
        self.render_gan_image = False
        self.current_colormap = 0
        self.colormap_dict = dict(colormaps)
        self.colormaps_names = [key for key, _ in colormaps]
        self.invert = False

        self.resolution_input = InputInt(viz, "resolution", value=512, add_to_args=True)
        self.background_color_input = InputTensor(viz, "background_color", value=[1.0, 1.0, 1.0], add_to_args=True)
        self.img_normalize_checkbox = CheckboxInput(viz, "img_normalize", value=False, add_to_args=True)
        self.invert_checkbox = CheckboxInput(viz, "invert", value=False, add_to_args=True)
        self.render_alpha_checkbox = CheckboxInput(viz, "render_alpha", value=False, add_to_args=True)
        self.render_depth_checkbox = CheckboxInput(viz, "render_depth", value=False, add_to_args=True)
        self.colormap_combo = Combo(viz, "colormap", self.colormaps_names)


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True, decoder=False):
        viz = self.viz
        if show:
            self.resolution_input()
            self.background_color_input()
            self.img_normalize_checkbox()
            self.invert_checkbox()
            alpha_changed = self.render_alpha_checkbox()
            depth_changed = self.render_depth_checkbox()
            self.colormap_combo()

            if self.render_alpha_checkbox.value and alpha_changed:
                self.render_depth_checkbox.value = False
            if self.render_depth_checkbox.value and depth_changed:
                self.render_alpha_checkbox.value = False

        viz.args.colormap = self.colormap_dict[self.colormap_combo.value]
