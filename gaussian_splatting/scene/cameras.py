from gaussian_splatting.gs_utils.graphics_utils import getProjectionMatrix


class CustomCam:
    def __init__(self, width, height, fovy, fovx, extr, znear=0.01, zfar=10):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self.world_view_transform = extr.T.inverse()
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = -self.world_view_transform[:3, 3]
