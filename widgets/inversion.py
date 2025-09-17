from copy import copy

import cv2
import dlib
from imgui_bundle import imgui, ImVec2
from imgui_bundle import portable_file_dialogs, immvision
import numpy as np
from PIL import Image

from splatviz_utils.dict_utils import EasyDict
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label, slider
from widgets.widget import Widget

from insightface import app


def get_crop_bound(lm):
    if len(lm) == 106:
        left_e = lm[38]
        right_e = lm[88]
        nose = lm[80]
        left_m = lm[52]
        right_m = lm[61]
        center = (lm[9] + lm[25]) * 0.5
    elif len(lm) == 68:
        left_e = np.mean(lm[36:42], axis=0)
        right_e = np.mean(lm[42:48], axis=0)
        nose = lm[33]
        left_m = lm[48]
        right_m = lm[54]
        center = (lm[0] + lm[16]) * 0.5
    else:
        raise ValueError(f"Unknown type of keypoints with a length of {len(lm)}")

    return [left_e, right_e, nose, left_m, right_m, center]


class KeypointDetectorInsightface:
    def __init__(self):
        self.app = app.FaceAnalysis(name='buffalo_s')  # enable detection model only
        self.app.prepare(ctx_id=0)

    def __call__(self, img):
        detection = self.app.get(img)
        if len(detection) > 0:
            return detection[0].landmark_2d_106
        return []

class KeypointDetectorDlib:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(f"{get_project_path()}/models/shape_predictor_68_face_landmarks.dat")

    def __call__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = self.detector(gray, 1)
        if len(rect) > 0:
            shape = self.predictor(gray, rect[0])
            return np.array([[p.x, p.y] for p in shape.parts()])
        return []

class Hyperparams:
    def __init__(self, name, default_value, min_value, max_value, log, dtype):
        self.name = name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.log = log
        self.dtype = dtype



class InversionWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Inversion")
        self.files_to_preprocess = []
        self.loaded_images = []
        self.run_inversion = False
        self.run_tuning = False
        self.loss_settings_inversion = EasyDict(
            id_loss=Hyperparams("ID Similarity", 0.1, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            mse_loss=Hyperparams("MSE", 0.01, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            lpips_loss=Hyperparams("LPIPS", 1.0, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            lr=Hyperparams("Learning Rate", 0.005, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            batch_size=Hyperparams("Batch Size", 1, min_value=1, max_value=32, log=False, dtype="int"),

        )
        self.loss_settings_tuning = EasyDict(
            id_loss=Hyperparams("ID Similarity", 0.1, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            mse_loss=Hyperparams("MSE", 0.01, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            lpips_loss=Hyperparams("LPIPS", 1.0, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            lr=Hyperparams("Learning Rate", 0.0005, min_value=0.0, max_value=1.0, log=True, dtype="float"),
            batch_size=Hyperparams("Batch Size", 1, min_value=1, max_value=32, log=False, dtype="int"),
        )
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.keypoint_detector = KeypointDetectorInsightface()
        # self.keypoint_detector_dlib = KeypointDetectorDlib()
        self.keypoints = []

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text("1. Load Images\n")
            if imgui.button("Open Images from Files", ImVec2(viz.label_w_large, 0)):
                files_from_dialog = portable_file_dialogs.open_file("Select Image", "./", filters=[], options=portable_file_dialogs.opt.multiselect).result()
                for file in files_from_dialog:
                    self.files_to_preprocess.append(np.array(Image.open(file).convert("RGB")))
                    self.loaded_images.append(np.array(Image.open(file).convert("RGB")))

            imgui.new_line()
            if imgui.collapsing_header("Webcam"):
                ret, frame = self.webcam.read()

                zoom = 200
                frame = frame[zoom:-zoom, 420 + zoom:-420-zoom, :]
                img_without_annotation = copy(frame[:, ::-1, ::-1])
                self.keypoints = self.keypoint_detector(frame)
                # keypoints_dlib = self.keypoint_detector_dlib(frame)
                # frame[self.keypoints.astype(int)[:, 1], self.keypoints.astype(int)[:, 0], :] = [0, 0, 255]

                for point in self.keypoints:
                    cv2.circle(frame, tuple(point.astype(int)), 5, (0, 255, 0), cv2.FILLED)

                frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
                immvision.image_display_resizable(f"webcam_image", frame, ImVec2(frame.shape[1], frame.shape[0]), is_bgr_or_bgra=False, refresh_image=True)
                if imgui.button("Take Image", ImVec2(viz.label_w_large, 0)):
                    self.loaded_images.append(img_without_annotation)
                    self.files_to_preprocess.append(img_without_annotation)


            im_size = self.viz.pane_w // (len(self.loaded_images) + 1)
            for i, image in enumerate(self.loaded_images):
                aspect_ratio = image.shape[1] / image.shape[0]
                immvision.image_display_resizable(f"image_{i}", image, ImVec2(im_size, int(im_size / aspect_ratio)), is_bgr_or_bgra=False, refresh_image=False)
                imgui.same_line(spacing=0)

            imgui.new_line()
            imgui.text("\n2. Image Preprocessing")
            if imgui.button("Preprocess", ImVec2(viz.label_w_large, 0)):
                viz.args.inversion_images = copy(self.files_to_preprocess)
                self.files_to_preprocess = []

            for i, image in enumerate(viz.preprocessed_images):
                immvision.image_display_resizable(f"image_pre_{i}", np.ascontiguousarray(image), ImVec2(im_size, im_size), is_bgr_or_bgra=False, refresh_image=False)
                imgui.same_line(spacing=0)
            self.loss_settings_inversion["batch_size"].max_value = len(viz.preprocessed_images)
            self.loss_settings_tuning["batch_size"].max_value = len(viz.preprocessed_images)

            imgui.new_line()
            imgui.push_item_width(viz.label_w_large)
            imgui.text("\n3. Run Latent Inversion")
            for key in self.loss_settings_inversion.keys():
                label(self.loss_settings_inversion[key].name, width=viz.label_w_large)
                if self.loss_settings_inversion[key].dtype == "float":
                    changed, self.loss_settings_tuning[key].default_value = imgui.input_float(f"##{key}_inversion", self.loss_settings_tuning[key].default_value, format="%.5f")
                elif self.loss_settings_inversion[key].dtype == "int":
                    changed, self.loss_settings_inversion[key].default_value = imgui.input_int(
                        label=f"##{key}_inversion",
                        v=int(self.loss_settings_inversion[key].default_value)
                    )

                imgui.same_line()
                self.loss_settings_inversion[key].default_value = slider(
                    self.loss_settings_inversion[key].default_value,
                    f"##{key}_inversion_slider",
                    min=self.loss_settings_inversion[key].min_value,
                    max=self.loss_settings_inversion[key].max_value,
                    log=self.loss_settings_inversion[key].log,
                    format="%.5f" if self.loss_settings_inversion[key].dtype == "float" else "%.0f"
                )

            if imgui.button("Stop Inversion" if self.run_inversion else "Start Inversion", ImVec2(viz.label_w_large, 0)):
                self.run_inversion = not self.run_inversion


            imgui.text("\n4. Run Generator Tuning")
            for key in self.loss_settings_tuning.keys():
                label(self.loss_settings_tuning[key].name, width=viz.label_w_large)
                if self.loss_settings_inversion[key].dtype == "float":
                    changed, self.loss_settings_tuning[key].default_value = imgui.input_float(f"##{key}_tuning", self.loss_settings_tuning[key].default_value, format="%.5f")
                elif self.loss_settings_inversion[key].dtype == "int":
                    changed, self.loss_settings_tuning[key].default_value = imgui.input_int(
                        label=f"##{key}_tuning",
                        v=int(self.loss_settings_tuning[key].default_value)
                    )

                imgui.same_line()
                self.loss_settings_tuning[key].default_value = slider(
                    self.loss_settings_tuning[key].default_value,
                    f"##{key}_tuning_slider",
                    min=self.loss_settings_tuning[key].min_value,
                    max=self.loss_settings_tuning[key].max_value,
                    log=self.loss_settings_tuning[key].log,
                    format="%.5f" if self.loss_settings_tuning[key].dtype == "float" else "%.0f"
                )

            if imgui.button("Stop Tuning" if self.run_tuning else "Start Tuning", ImVec2(viz.label_w_large, 0)):
                self.run_tuning = not self.run_tuning


            # interface to backend
            viz.args.inversion_hyperparams = {key: self.loss_settings_inversion[key].default_value for key in self.loss_settings_inversion.keys()}
            viz.args.tuning_hyperparams = {key: self.loss_settings_tuning[key].default_value for key in self.loss_settings_tuning.keys()}
            viz.renderer.update_all_the_time = self.run_inversion or self.run_tuning
            viz.args.run_inversion = self.run_inversion
            viz.args.run_tuning = self.run_tuning



