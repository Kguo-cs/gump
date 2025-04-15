import logging
import os
import warnings
from typing import Optional
import torch.multiprocessing

import hydra
import torch
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import \
    build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.utils.utils_config import \
    update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from nuplan.planning.training.experiments.training import (
    TrainingEngine, build_training_engine)
from omegaconf import DictConfig, OmegaConf
import cv2
import os
from hydra import initialize, compose


# os.environ["NUPLAN_DEVKIT_PATH"] = "/home/ke/code/GUMP/third_party/nuplan-devkit"
# os.environ["NUPLAN_DATA_ROOT"] = "/home/ke/code/GUMP/nuplan_data/dataset/nuplan-v1.1/splits/mini"
# os.environ["NUPLAN_MAPS_ROOT"] = "/home/ke/code/GUMP/nuplan_data/dataset/maps"
# os.environ["NUPLAN_EXP_ROOT"] = "/home/ke/code/GUMP"
#

# Environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

gump_path='~/scratch/keguo_projects/gump' #'/home/ke/code/GUMP'#~/scratch/keguo_projects/gump'

os.environ["NUPLAN_DEVKIT_PATH"] =gump_path+ "/third_party/nuplan-devkit"
os.environ["NUPLAN_DATA_ROOT"] = gump_path+"/nuplan_data/dataset/nuplan-v1.1/splits/train"
os.environ["NUPLAN_MAPS_ROOT"] =gump_path+ "/nuplan_data/dataset/maps"
os.environ["NUPLAN_EXP_ROOT"] = gump_path

# Config paths
SAVE_DIR = "./workspace/test/"
EXPERIMENT = "test_nuplan"
CACHE_DIR = gump_path+"/home/ke/code/GUMP/save_dir/caching/caching/2025.04.12.12.02.48/cache_dir_v1_1"
CACHE_META_PATH = f"{CACHE_DIR}/metadata/cache_dir_v1_1_metadata_node_0.csv"
DATA_ROOT= gump_path+"/nuplan_data/dataset/nuplan-v1.1/splits/train"
MAP_ROOT=gump_path+"/nuplan_data/dataset/maps/"

cv2.setNumThreads(1)
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')
logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and
# experiment paths
set_default_path()

# Add a new resolver that supports eval
OmegaConf.register_new_resolver("eval", eval)

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config')

CONFIG_NAME = os.getenv('DEFAULT_CONFIG', 'horizon_training')


# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    # Build plugins (compatible with mmdet)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_path = _module_dir.replace("/", ".")
            if _module_path.startswith("."):
                _module_path = _module_path[1:]
            logger.info(f"Plugin directory: {_module_path}")
            plg_lib = importlib.import_module(_module_path)

    if cfg.py_func == 'train':
        # Build training engine
        engine = build_training_engine(cfg, worker)

        # Run training
        logger.info('Starting training...')
        
        # compatible with pytorch_lightning 1.3.8 and 2.1.11
        if hasattr(engine.trainer, 'resume_from_checkpoint'):
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule, ckpt_path=engine.trainer.resume_from_checkpoint)
        else:
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'test':
        # Build training engine
        engine = build_training_engine(cfg, worker)

        # Test model
        logger.info('Starting testing...')

        my_ckpt_path = cfg.checkpoint.ckpt_path
        assert isinstance(my_ckpt_path, str), 'Checkpoint path must be a string'
        assert os.path.exists(my_ckpt_path), f'Checkpoint path {my_ckpt_path} does not exist'
        
        my_ckpt = torch.load(my_ckpt_path, map_location='cpu')
        engine.model.load_state_dict(my_ckpt['state_dict']) 

        engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache':
        # Precompute and cache all features
        cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


# Initialize Hydra with config path (point to your config dir, adjust as needed)
with initialize( config_path=CONFIG_PATH):
    # cfg = compose(
    #     config_name=CONFIG_NAME,
    #     overrides=[
    #         f"group={SAVE_DIR}",
    #         f"experiment_name={EXPERIMENT}",
    #         "py_func=train",
    #         "seed=0",
    #         "scenario_builder=nuplan",
    #         f"scenario_builder.data_root={os.environ['NUPLAN_DATA_ROOT']}",
    #         "scenario_builder.scenario_mapping.subsample_ratio_override=1",
    #         "lightning.trainer.params.accelerator=gpu",
    #         "lightning.trainer.params.max_epochs=15",
    #         "lightning.trainer.params.max_time=14:32:00:00",
    #         "lightning.trainer.params.precision=16",
    #         "lightning.trainer.params.gradient_clip_val=5.0",
    #         "lightning.trainer.params.strategy=ddp_find_unused_parameters_true",
    #         "lightning.trainer.params.accumulate_grad_batches=2",
    #         "+lightning.trainer.params.val_check_interval=1.0",
    #         "data_loader.params.batch_size=1",
    #         "data_loader.params.num_workers=8",
    #         "worker=single_machine_thread_pool",
    #         "model=gump_nuplan_lamma_sm_v1_1",
    #         "optimizer=adamw",
    #         "optimizer.lr=1e-4",
    #         "optimizer.weight_decay=1e-3",
    #         "lr_scheduler=multistep_lr",
    #         "lr_scheduler.milestones=[8,13]",
    #         "lr_scheduler.gamma=0.2",
    #         "lightning.trainer.checkpoint.resume_training=false",
    #         "scenario_filter=all_scenarios",
    #         "+checkpoint.ckpt_path=null",
    #         "+checkpoint.strict=True",
    #         "+checkpoint.resume=False",
    #         f"cache.cache_path={CACHE_DIR}",
    #         f"cache.cache_metadata_path={CACHE_META_PATH}",
    #         "cache.force_feature_computation=false",
    #         "cache.use_cache_without_dataset=true",
    #         "cache.versatile_caching=false",
    #         "+training=training_nuplan_gump_v1_1",
    #         # Optional: Uncomment if you want to resume from a checkpoint
    #         # "+checkpoint.ckpt_path='/mnt/nas26/yihan01.hu/tmp/epoch_llama_sm.ckpt'",
    #     ],
    # )
    cfg = compose(
        config_name=CONFIG_NAME,
        overrides=[
            f"group={SAVE_DIR}",
            f"experiment_name={EXPERIMENT}",
            "py_func=cache",
            "job_name=caching",
            "scenario_builder=nuplan",
            "scenario_filter=all_scenarios",
            "scenario_builder.scenario_mapping.subsample_ratio_override=1",
            "worker=single_machine_thread_pool",
            "worker.use_process_pool=false",
            "worker.max_workers=1",
            "model=gump_nuplan_gptbase_v1_1",
            f"scenario_builder.data_root={DATA_ROOT}",
            f"scenario_builder.map_root={MAP_ROOT}",
            "scenario_filter.timestamp_threshold_s=15",
            "scenario_filter.num_scenarios_per_type=1000",
            "scenario_filter.expand_scenarios=false",
            "scenario_filter.remove_invalid_goals=true",
            f"cache.cache_path={CACHE_DIR}",
            "cache.force_feature_computation=true",
            "cache.versatile_caching=false",
            "+training=training_nuplan_gump_v1_1",
        ],
    )
    print(OmegaConf.to_yaml(cfg))  # Optional: See the final config
    main(cfg)

# if __name__ == '__main__':
#     main()
