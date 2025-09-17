import os
from imgui_bundle import imgui
from splatviz_utils.gui_utils import imgui_utils
from widgets.widget import Widget


class LoadWidget(Widget):
    def __init__(self, viz, file_ending):
        super().__init__(viz, "Load")
        self.filter = ""
        self.file_ending = file_ending
        self.ply = self.items[0]

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            _changed, self.filter = imgui.input_text("Filter", self.filter)
            if imgui_utils.button("Browse", width=viz.button_w, enabled=True):
                imgui.open_popup("browse_pkls_popup")

            if imgui.begin_popup("browse_pkls_popup"):
                for item in self.items:
                    clicked = imgui.menu_item_simple(os.path.relpath(item, self.root))
                    if clicked:
                        self.ply = item
                imgui.end_popup()

            imgui.same_line()
            imgui.text(self.ply)
        viz.args.ply_file_paths = [self.ply]
        viz.args.current_ply_names = self.ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
