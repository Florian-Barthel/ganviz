#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from gsplat import rasterization

from gaussian_splatting.scene.gaussian_model import GaussianModel


def render_simple(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    debug=False,
):
    """Render one camera view and return tensors in the legacy renderer layout."""
    del debug  # gsplat does not expose the old rasterizer's debug flag.

    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling * scaling_modifier
    opacities = pc.get_opacity.squeeze(-1)
    colors = pc.get_features if override_color is None else override_color
    sh_degree = pc.active_sh_degree if override_color is None else None

    assert means.ndim == 2 and means.shape[-1] == 3, means.shape
    assert quats.shape == (means.shape[0], 4), quats.shape
    assert scales.shape == (means.shape[0], 3), scales.shape
    assert opacities.shape == (means.shape[0],), opacities.shape
    assert colors.shape[0] == means.shape[0], colors.shape
    assert bg_color.shape == (3,), bg_color.shape

    # gsplat treats cameras as a distinct dimension. The Gaussians remain
    # unbatched, while this adapter explicitly renders one camera.
    rendered, rendered_alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewpoint_camera.world_to_cam_transform.unsqueeze(0),
        Ks=viewpoint_camera.intrinsics.unsqueeze(0),
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        near_plane=viewpoint_camera.znear,
        far_plane=viewpoint_camera.zfar,
        sh_degree=sh_degree,
        packed=False,
        backgrounds=bg_color.unsqueeze(0),
        render_mode="RGB+D",
    )

    radii = meta["radii"][0]
    viewspace_points = meta["means2d"][0]
    try:
        viewspace_points.retain_grad()
    except RuntimeError:
        pass

    # gsplat returns [camera, height, width, channel]. Keep the established
    # renderer contract used by the GUI: [channel, height, width].
    rendered_image = rendered[0, ..., :3].permute(2, 0, 1)
    rendered_depth = rendered[0, ..., 3:].permute(2, 0, 1)
    rendered_alpha = rendered_alpha[0].permute(2, 0, 1)

    return {
        "render": rendered_image,
        "viewspace_points": viewspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "alpha": rendered_alpha,
        "depth": rendered_depth,
    }
