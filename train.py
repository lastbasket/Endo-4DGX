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

import numpy as np
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss, Exp_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import cv2
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from torchmetrics.functional.regression import pearson_corrcoef
from utils import helper
from utils.resnet_swag import content_loss
from torchvision import transforms, models


import lpips
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# resnet = models.resnet50(pretrained=True)
# resnet.to('cuda').eval().float()


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration, stage=stage)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            # idx = 0
            idx = randint(0, len(viewpoint_stack)-1)
            # idx = iteration % len(viewpoint_stack)
        else:
            idx = randint(0, len(viewpoint_stack)-1)
            # idx = iteration % len(viewpoint_stack)
            
        viewpoint_cams = [viewpoint_stack[idx]]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        images_concealing = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], \
                    render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            
            image_concealing = render_pkg['render_restored']
            images_concealing.append(image_concealing.unsqueeze(0))
            images.append(image.unsqueeze(0))
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if gt_depth.ndim == 2:
                gt_depth = gt_depth.unsqueeze(0)
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images,0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images,0)
        gt_depths = torch.cat(gt_depths, 0)
        images_concealing = torch.cat(images_concealing, 0)
        masks = torch.cat(masks, 0)
        
        
        rendered_images_flat = rendered_images.permute(0,2,3,1).reshape(-1, 3)
        gt_images_flat = gt_images.permute(0,2,3,1).reshape(-1, 3)
        images_concealing_flat = images_concealing.permute(0,2,3,1).reshape(-1, 3)
        masks_flat = masks.permute(0,2,3,1).reshape(-1, 1).repeat(1, 3)
        
        mean_rgb_fine = torch.mean(rendered_images_flat[masks_flat].reshape(-1, 3), dim=0)

        Ll1 = l1_loss(images_concealing, gt_images, masks)
        loss_control = helper.Exp_loss_global(mean_val=hyper.eta)((rendered_images_flat[masks_flat].reshape(-1, 3)))
        # loss_control = Exp_loss(patch_size=64, mean_val=hyper.eta)((rendered_images))
        
        # loss_control = content_loss(rendered_images, viewpoint_cam.reference[None].cuda(), resnet)
        if viewpoint_cam.illu_type == 'low_light':
            loss_structure = helper.Structure_Loss(contrast=hyper.eta * hyper.con/10)(gt_images_flat[masks_flat].reshape(-1, 3), \
                rendered_images_flat[masks_flat].reshape(-1, 3))
        else:
            loss_structure = helper.Structure_Loss(contrast=hyper.con)(rendered_images_flat[masks_flat].reshape(-1, 3), \
                gt_images_flat[masks_flat].reshape(-1, 3))
        loss_cc = helper.colour(mean_rgb_fine)
        # print('depths', rendered_depths.shape)
        # print('gt', gt_depths.shape)
        depth_loss = opt.depth_weight * l1_loss(torch.clamp(rendered_depths/(rendered_depths.max()+1e-6), 0, 1), \
            torch.clamp(gt_depths/(gt_depths.max()+1e-6), 0, 1), masks, True)
        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = opt.tv_weight * (img_tvloss + depth_tvloss)
        
        loss = Ll1 + depth_loss + tv_loss + opt.control_weight*loss_control #+ 1e-1*loss_structure + 1e-6*loss_cc
        
        
        # out_save_dep = rendered_depths.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        # out_save_dep = np.clip(out_save_dep/(out_save_dep.max()+1e-6), 0, 1)
        # out_save_dep = (out_save_dep*255).astype(np.uint8)
        # cv2.imwrite('out_dep.png', out_save_dep)
        
        # if (iteration+1)%10 == 0:
            # out_save = rendered_images.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            # out_save = cv2.cvtColor((np.clip(out_save, 0, 1)*255), cv2.COLOR_RGB2BGR).astype(np.uint8)
            # cv2.imwrite('out.png', out_save)
            
            # out_save_con = images_concealing.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            # out_save_con = cv2.cvtColor((np.clip(out_save_con, 0, 1)*255), cv2.COLOR_RGB2BGR).astype(np.uint8)
            # cv2.imwrite('out_con.png', out_save_con)

        psnr_ = psnr(images_concealing, gt_images, masks).mean().double()        
        
        loss.backward()
        for param in gaussians.get_param():
            torch.nn.utils.clip_grad_value_(param, clip_value=0.5)
            # torch.nn.utils.clip_grad_norm_(param, max_norm=1)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                string_dict = {"Loss": f"{ema_loss_for_log:.{7}f}",
                                "psnr": f"{psnr_:.{2}f}",
                                "point":f"{total_point}"}
                # if stage == "fine" and hyper.time_smoothness_weight != 0:
                progress_bar.set_postfix(string_dict)
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            metrics = {'Ll1':Ll1, 'depth_loss':depth_loss, 'elapsed':iter_start.elapsed_time(iter_end), \
                'psnr':psnr_, 'control_loss':loss_control, 'structure_loss':loss_structure, 'cc_loss':loss_cc}
            
            training_report(tb_writer, iteration, metrics, \
                testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - \
                        opt.opacity_threshold_fine_after)/(opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - \
                        opt.densify_grad_threshold_after)/(opt.densify_until_iter )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # print('densify', iteration, opt.densification_interval, opt.densify_from_iter)
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    # print('pruning', iteration, opt.pruning_interval, opt.pruning_from_iter)
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    if not args.no_fine:
        # gaussians.update_lr('illumination_embeddings', 1e-2)
        # gaussians.update_lr('region', 1e-2)
        # gaussians.update_lr('spatial', 1e-3)
        
        gaussians.update_lr('illumination_embeddings', opt.illumination_embedding_lr_fine)
        gaussians.update_lr('region', opt.region_lr_fine)
        gaussians.update_lr('spatial', opt.spatial_lr_fine)
        
        # gaussians.update_lr('xyz', opt.position_lr_init * self.spatial_lr_scale)
        
        # opt.pruning_interval=200
        # opt.pruning_from_iter=1000
        # opt.densification_interval=200
        # opt.densify_from_iter=1000
        
        opt.pruning_interval=opt.pruning_interval_fine
        opt.pruning_from_iter=opt.pruning_from_iter
        opt.densification_interval=opt.densification_interval_fine
        opt.densify_from_iter=opt.densify_from_iter_fine
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, metrics, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/l1_loss', metrics['Ll1'].item(), iteration)
        tb_writer.add_scalar(f'{stage}/depth_loss', metrics['depth_loss'].item(), iteration)
        tb_writer.add_scalar(f'{stage}/psnr', metrics['psnr'].item(), iteration)
        tb_writer.add_scalar(f'{stage}/control_loss', metrics['control_loss'].item(), iteration)
        tb_writer.add_scalar(f'{stage}/structure_loss', metrics['structure_loss'].item(), iteration)
        tb_writer.add_scalar(f'{stage}/cc_loss', metrics['cc_loss'].item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', metrics['elapsed'], iteration)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000,2000, 3000,4000, 5000, 6000, 7000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")
