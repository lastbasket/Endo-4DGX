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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
    override_color = None, stage="fine", embedding_idx=-1, embedding=None, illu_type=None, time=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
        
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    if time is None:
        # print('time', viewpoint_camera.time)
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        # print('use pre_time', time.item())
        time = time.to(means3D.device).repeat(means3D.shape[0],1)
    means2D = screenspace_points
    opacity = pc._opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    
    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point])
    # print(time.max())
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity)

    shs = None
    colors_precomp = pc.features
    
    if embedding is not None:
        app_embeddings = embedding
    else:
        if embedding_idx == -1:
            app_embeddings = pc.get_embedding(viewpoint_camera.id)[None]
        else:
            app_embeddings = pc.get_embedding(embedding_idx)[None]
            
    if illu_type is None:
        input_illu_type=viewpoint_camera.illu_type
    else:
        input_illu_type=illu_type
        
    color_tone = pc.region(colors_precomp, app_embeddings.repeat(len(means2D), 1), \
        ill_type=input_illu_type)
    
    # for ablation
    # color_tone = pc.region(colors_precomp, \
    #     ill_type=input_illu_type)
    
    # color_tone = pc.region(colors_precomp, app_embeddings.repeat(len(means2D), 1), \
    #     ill_type='over_exposure')
    # app_embeddings = pc.get_embedding(6)[None].repeat(len(means2D), 1)
    # color_tone = pc.region(torch.cat((colors_precomp, app_embeddings), dim=-1), \
    #     ill_type='over_exposure')#low_light
    
    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None, #shs*pc.get_concealing[:, None, :]
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # for ablation
    # rendered_image_concealing = rendered_image
    
    rendered_image_concealing, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None, #shs*pc.get_concealing[:, None, :]
        colors_precomp = color_tone,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    # for ablation
    # rendered_image_concealing = pc.spatial(rendered_image_concealing.permute(1,2,0)).permute(2,0,1)
    
    rendered_image_concealing = pc.spatial(rendered_image_concealing, \
        app_embeddings)
    
    return {"render": rendered_image,
            "render_restored":rendered_image_concealing,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,}

