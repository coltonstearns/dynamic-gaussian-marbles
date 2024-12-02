from nerfstudio.configs.base_config import ViewerConfig
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
                outlier_std_ratio=2.0,
                text_guided=True,
            ),
            model=GaussianSplattingModelConfig(
                chamfer_agg_group_ratio=0.125,
                chamfer_loss_weight=0.2,
                delta_position_lr=0.002,
                depthmap_loss_weight=0.06,
                downsample_reducescale=0.85,
                frame_transport_dropout=0.5,
                freeze_frames_of_origin=True,
                freeze_previous_in_motion_estimation=True,
                instance_isometry_loss_weight=0.5,
                instance_isometry_numpairs=4096,
                isometry_knn=24,
                isometry_knn_radius=0.3,
                isometry_loss_weight=0.4,
                consistency_loss_weight=0.01,
                isometry_weight_background_factor=1.0,
                lpips_loss_weight=0.002,
                number_of_gaussians=300000,
                tracking_knn=32,
                tracking_loss_weight=0.025,
                tracking_radius=4,
                tracking_window=12,
                tracking_depth_loss_weight=1.0,
                photometric_loss_weight=0.2,
                prune_points=True,
                scaling_loss_weight=3.0,
                segmentation_loss_weight=0.4,
                static_background=False,
                supervision_downscale=1,
                velocity_smoothing_loss_weight=0,
            ),
            stages=[
                {'type': 'global-adjust', 'steps': 64},
                {'type': 'motion-estimation-and-merge', 'steps': 160},
                {'type': 'global-adjust', 'steps': 64},
                {'type': 'motion-estimation-and-merge', 'steps': 160},
                {'type': 'global-adjust', 'steps': 64},
                {'type': 'motion-estimation-and-merge', 'steps': 160},
                {'type': 'global-adjust', 'steps': 64},
                {'type': 'motion-estimation-and-merge', 'steps': 160},
                {'type': 'global-adjust', 'steps': 64},
                {'type': 'motion-estimation-and-merge', 'steps': 160},
                {'type': 'global-adjust', 'steps': 64},
                {'type': 'expand-motion-sliding-window', 'steps': 128, 'window-size': 16},
                {'type': 'global-adjust', 'steps': 64},
                # {'type': 'motion-estimation-and-merge', 'steps': 160},
                # {'type': 'global-adjust', 'steps': 64},
                # {'type': 'motion-estimation-and-merge', 'steps': 160},
                # {'type': 'global-adjust', 'steps': 64},
                # {'type': 'upsample', 'factor': 2},
                # {'type': 'global-adjust', 'steps': 64},
                {'type': 'global-adjust', 'steps': 64},
            ],
            text_guided=True,
            prompt="change the person to a clown and don't change his height. don't change anything else in the scene.",
            gs_steps=2500

        ),
        optimizers={
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15,
                            quit_on_train_completion=True),
        vis="viewer",
    )