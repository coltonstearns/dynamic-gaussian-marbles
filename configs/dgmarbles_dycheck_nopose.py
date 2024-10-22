from nerfstudio.configs.base_config import ViewerConfig
# from nerfstudio.engine.trainer import TrainerConfig
from src.models.trainer import TrainerConfig
from src.data.dycheck_dataparser import DycheckDataParserConfig
from src.models.pipeline import GaussianSplattingPipelineConfig
from src.models.model import GaussianSplattingModelConfig
from src.data.datamanager import GaussianSplattingDataManagerConfig, GaussianSplattingDataManager

train_config = \
    TrainerConfig(
        method_name="gsplatting",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        mixed_precision=True,
        pipeline=GaussianSplattingPipelineConfig(
            datamanager=GaussianSplattingDataManagerConfig(
                _target=GaussianSplattingDataManager,
                dataparser=DycheckDataParserConfig(
                    downscale_factor=2
                ),
                znear=0.01,
                zfar=20.0,
                depth_remove_outliers=True,
                outlier_std_ratio=1.5
            ),
            model=GaussianSplattingModelConfig(
                chamfer_agg_group_ratio=0.25,
                chamfer_loss_weight=0.0,
                delta_position_lr=0.002,
                depthmap_loss_weight=0.1,
                downsample_reducescale=0.85,
                frame_transport_dropout=0.5,
                freeze_frames_of_origin=True,
                freeze_previous_in_motion_estimation=True,
                instance_isometry_loss_weight=1.7,
                instance_isometry_numpairs=4096,
                isometry_knn=24,
                isometry_knn_radius=0.3,
                isometry_loss_weight=0.25,
                isometry_weight_background_factor=1.0,
                lpips_loss_weight=0.001,
                number_of_gaussians=220000,
                tracking_knn=32,
                tracking_loss_weight=0.07,
                tracking_radius=4,
                tracking_window=12,
                tracking_depth_loss_weight=0.0,
                photometric_loss_weight=0.3,
                prune_points=True,
                scaling_loss_weight=6,
                segmentation_loss_weight=0.4,
                static_background=False,
                supervision_downscale=1,
                velocity_smoothing_loss_weight=0,
            ),
            stages=[
                # expand both foreground and background
                {'type': 'global-adjust', 'steps': 32},
                {'type': 'motion-estimation-and-merge', 'steps': 80},
                {'type': 'global-adjust', 'steps': 32},
                {'type': 'motion-estimation-and-merge', 'steps': 80},
                {'type': 'global-adjust', 'steps': 32},
                {'type': 'motion-estimation-and-merge', 'steps': 80},
                {'type': 'global-adjust', 'steps': 32},
                {'type': 'expand-motion-sliding-window', 'steps': 80, 'window-size': 4, 'updating': 'foreground'},
                {'type': 'global-adjust', 'steps': 32, 'updating': 'foreground'},

                # only expand and update background
                {'type': 'motion-estimation-and-merge', 'steps': 80, 'updating': 'background'},
                {'type': 'global-adjust', 'steps': 32, 'updating': 'background'},
                {'type': 'motion-estimation-and-merge', 'steps': 80, 'updating': 'background'},
                {'type': 'global-adjust', 'steps': 32, 'updating': 'background'},
                {'type': 'expand-motion-sliding-window', 'steps': 80, 'window-size': 16, 'updating': 'background'},
                {'type': 'global-adjust', 'steps': 32, 'updating': 'background'},

            ]
        ),
        optimizers={
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15,
                            quit_on_train_completion=True),
        vis="viewer",
    )
