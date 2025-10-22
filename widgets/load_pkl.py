from imgui_bundle import imgui, ImVec2
from imgui_bundle._imgui_bundle import portable_file_dialogs

from splatviz_utils.gui_utils import imgui_utils
from widgets.widget import Widget


class LoadWidget(Widget):
    def __init__(self, viz, file_ending):
        super().__init__(viz, "Load")
        self.filter = ""
        self.file_ending = file_ending
        self.ply = ""

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        if show:
            if imgui.button("Load pkl", ImVec2(self.viz.label_w_large, 0)):
                files_from_dialog = portable_file_dialogs.open_file("Select pkl", self.viz.gan_path, filters=["All Files", "*.pkl"]).result()
                if len(files_from_dialog) > 0:
                    self.ply = files_from_dialog[0]

        self.viz.args.ply_file_paths = [self.ply]
        self.viz.args.current_ply_names = self.ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
