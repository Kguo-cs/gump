#! /usr/bin/env bash
EXPERIMENT="caching"

SAVE_DIR="./save_dir/"
CACHE_DIR="./cache_dir_v1_1/"

DATA_ROOT=/home/ke/code/GUMP/nuplan_data/dataset/nuplan-v1.1/splits/mini
MAP_ROOT=/home/ke/code/GUMP/nuplan_data/dataset/maps/
export NUPLAN_DEVKIT_PATH=/home/ke/code/GUMP/third_party/nuplan-devkit
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONPATH=$NUPLAN_DEVKIT_PATH:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 # This is to avoid OpenBlas creating too many threads
export OMP_NUM_THREADS=1  # Control the number of threads per process for OpenMP

python nuplan_extent/planning/script/run_training.py \
    group=$SAVE_DIR \
    cache.cache_path=$CACHE_DIR \
    cache.force_feature_computation=true \
    cache.versatile_caching=false \
    experiment_name=$EXPERIMENT \
    job_name=caching \
    py_func=cache \
    +training=training_nuplan_gump_v1_1 \
    scenario_builder=nuplan \
    scenario_filter=all_scenarios \
    scenario_builder.scenario_mapping.subsample_ratio_override=1 \
    worker=single_machine_thread_pool \
    worker.use_process_pool=false \
    worker.max_workers=1 \
    model=gump_nuplan_gptbase_v1_1 \
    scenario_builder.data_root=$DATA_ROOT \
    scenario_builder.map_root=$MAP_ROOT \
    scenario_filter.timestamp_threshold_s=15 \
    scenario_filter.num_scenarios_per_type=10 \
    scenario_filter.expand_scenarios=false \
    scenario_filter.remove_invalid_goals=true 
