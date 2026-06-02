import torch

from gaussian_splatting.gs_utils.graphics_utils import fov2focal, getProjectionMatrix


class CustomCam:
    def __init__(self, width, height, fovy, fovx, extr, znear=0.01, zfar=10):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self.world_to_cam_transform = extr.inverse()
        self.world_view_transform = self.world_to_cam_transform.T
        self.intrinsics = torch.tensor(
            [
                [fov2focal(self.FoVx, self.image_width), 0.0, self.image_width / 2],
                [0.0, fov2focal(self.FoVy, self.image_height), self.image_height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=extr.dtype,
            device=extr.device,
        )
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = extr[:3, 3]
