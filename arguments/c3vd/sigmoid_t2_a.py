ModelParams = dict(
    extra_mark = 'c3vd',
    camera_extent = 10,
    mode ='monocular'
)

OptimizationParams = dict(
    coarse_iterations = 3000,
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.0016,
    grid_lr_final = 0.000016,
    iterations = 7000,
    percent_dense = 0.016,
    opacity_reset_interval = 7000,
    position_lr_max_steps = 7000,
    
    # region_lr = 0.01,
    # spatial_lr = 0.001, # 0.01
    # illumination_embedding_lr= 0.001, # 0.01
    
    # region_lr_fine = 0.001, # 0.001,
    # spatial_lr_fine = 0.0001, #0.0001,
    # illumination_embedding_lr_fine = 0.001,
    
    region_lr = 0.01,
    spatial_lr = 0.001, # 0.01
    illumination_embedding_lr= 0.001, # 0.01
    
    region_lr_fine = 0.01, # 0.001,
    region_lr_fine_final = 0.001, # 0.001,
    
    spatial_lr_fine = 0.001, #0.0001,
    spatial_lr_fine_final = 0.0001, #0.0001,
    
    illumination_embedding_lr_fine = 0.001,
    illumination_embedding_lr_fine_final = 0.0001,
    
    pruning_interval_fine=100, #200,
    pruning_from_iter_fine=500, #1000,
    densification_interval_fine=100, #200,
    densify_from_iter_fine=500, #1000,
    
    # pruning_interval_fine=200,
    # pruning_from_iter_fine=1000,
    # densification_interval_fine=200,
    # densify_from_iter_fine=1000,
    
    control_weight = 1,
    depth_weight=1e-2,
    tv_weight=1e-2,
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 64,
     'resolution': [64, 64, 64, 100]

    },
    multires = [1, 2, 4, 8],
    defor_depth = 0,
    net_width = 32,
    plane_tv_weight = 0,
    time_smoothness_weight = 0,
    l1_time_planes =  0,
    weight_decay_iteration=0,
    eta = 0.6,
    con = 12
)