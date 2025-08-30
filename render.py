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
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, OptimizationParams
from scene import GaussianModel
from time import time
import open3d as o3d
from utils.graphics_utils import fov2focal
import json


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, reconstruct=False, embeddings=None,
               embedding_idx=-1, illu_type=None, wo_restoration=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_restored_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_restored")
    
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
    pcd_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pcd")
    pcd_restored_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pcd_restored")

    makedirs(render_path, exist_ok=True)
    makedirs(render_restored_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    makedirs(pcd_path, exist_ok=True)
    makedirs(pcd_restored_path, exist_ok=True)
    
    
    with open(os.path.join(model_path, 'embedding_info.json'), 'r', encoding='utf-8') as file:
        embedding_dict = json.load(file)
    
    render_images = []
    render_depths = []
    render_images_restored = []
    gt_list = []
    gt_depths = []
    mask_list = []

    test_times = 1
    
    time_view=None
    embedding=None
    # illu_type = 'low_light' #cutting 'over_exposure' #pulling
    # 
    if illu_type is None and embedding_idx != -1:
        for key, sub_dict in embedding_dict:
            if sub_dict['train_count'] == embedding_idx:
                illu_type = sub_dict['illu_type']
    
    for i in range(test_times):
        for idx, view in enumerate(views):
            if idx == 0 and i == 0:
                time1 = time()
            if embedding_idx==-1 and illu_type is None:
                if wo_restoration:
                    # varying illumination reconstruction, use the previous neighbour
                    embedding_idx = embedding_dict[str(view.uid-1)]['train_count']
                    embedding = embeddings[view.id][None]
                    illu_type = None
                    
                    # embedding=None
                    # embedding_idx = 7
                    # illu_type = 'low_light' #cutting 'over_exposure' #pulling 
                    # print('time_view', time_view)
                    # print('json embedding_idx', embedding_idx)

                else:
                    # default testing restoration reference
                    embedding_idx = 7
                    illu_type = 'low_light' if "cutting" in model_path else 'over_exposure' #pulling
                    time_view=None
            
            rendering = render(view, gaussians, pipeline, background, embedding_idx=embedding_idx, \
                embedding=embedding, illu_type=illu_type, time=time_view)
            
            if i == test_times-1:
                render_depths.append(rendering["depth"])
                render_images.append(rendering["render"].cpu())
                render_images_restored.append(rendering["render_restored"].cpu())
                
                if name in ["train", "test", "video"]:
                    gt = view.original_image[0:3, :, :]
                    gt_list.append(gt)
                    mask = view.mask
                    mask_list.append(mask)
                    gt_depth = view.depth
                    gt_depths.append(gt_depth)

    time2=time()
    print("FPS:",(len(views))*test_times/(time2-time1))
    # import pdb; pdb.set_trace()
    count = 0
    print("writing images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
            
    count = 0
    print("writing rendering images restored.")
    if len(render_images_restored) != 0:
        for image in tqdm(render_images_restored):
            torchvision.utils.save_image(image, os.path.join(render_restored_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image = image.float() / image.max()
            torchvision.utils.save_image(image, os.path.join(depth_path, '{0:05d}'.format(count) + ".png"))
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image = image.float() / image.max()
            torchvision.utils.save_image(image, os.path.join(gt_depth_path, '{0:05d}'.format(count) + ".png"))
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video_wo_restoration.mp4'), render_array, fps=30, quality=8)
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    
    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    
    if reconstruct:
        reconstruct_point_cloud(render_images, mask_list, render_depths, camera_parameters, name, pcd_path)
        reconstruct_point_cloud(render_images_restored, mask_list, render_depths, camera_parameters, name, pcd_restored_path)

def render_sets(dataset : ModelParams, optimization, hyperparam, iteration : int, pipeline : PipelineParams, \
    skip_train : bool, skip_test : bool, skip_video: bool, pc: bool, embedding_idx=-1, illu_type=None, wo_restoration=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, reconstruct=pc,
                       embedding_idx=embedding_idx, illu_type=illu_type, wo_restoration=wo_restoration)
        if not skip_test:
            # test_embeddings = gaussians.optimize_embeddings(scene.getTestCameras(), dataset, optimization, pipeline)
            test_embeddings = None
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, reconstruct=pc, \
                embeddings=test_embeddings, embedding_idx=embedding_idx, illu_type=illu_type, wo_restoration=wo_restoration)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, reconstruct=pc,
                       embedding_idx=embedding_idx, illu_type=illu_type, wo_restoration=wo_restoration)

def reconstruct_point_cloud(images, masks, depths, camera_parameters, name, pcd_path):
    import cv2
    import copy
    frames = np.arange(len(images))
    # frames = [0]
    focal_x, focal_y, width, height = camera_parameters
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()
        depth_np = depth_np.squeeze(0)
        mask = masks[i_frame]
        mask = mask.squeeze(0).cpu().numpy()
        
        rgb_new = copy.deepcopy(rgb_np)

        depth_smoother = (128, 64, 64)
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
        
        close_depth = np.percentile(depth_np[depth_np!=0], 5)
        inf_depth = np.percentile(depth_np, 95)
        depth_np = np.clip(depth_np, close_depth, inf_depth)

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, width / 2),
            project_valid_depth_only=True
        )
        pc_path = os.path.join(pcd_path, 'frame_{}.ply'.format(i_frame))
        print('pcd:', pc_path)
        o3d.io.write_point_cloud(pc_path, pcd)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--embedding_idx", default=-1, type=int)
    parser.add_argument("--illumination_type", default=None, type=str)
    parser.add_argument("--wo_restoration", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pc", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), 
                op.extract(args), 
                hyperparam.extract(args), 
                args.iteration, 
                pipeline.extract(args), 
                args.skip_train, 
                args.skip_test,
                args.skip_video,
                args.pc,
                args.embedding_idx,
                args.illumination_type,
                args.wo_restoration)