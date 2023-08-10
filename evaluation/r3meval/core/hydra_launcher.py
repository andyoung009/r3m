# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time as timer
import hydra
import multiprocessing
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append('/data/ML_document/r3m/evaluation')
print(sys.path)
from train_loop import bc_train_loop
# import train_loop

cwd = os.getcwd()

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="BC_config", config_path="config")
def configure_jobs(job_data:dict) -> None:
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    
    print("========================================")
    print("Job Configuration")
    print("========================================")

    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))

    job_data['cwd'] = cwd
    with open('job_config.json', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)
    print(OmegaConf.to_yaml(job_data))
    bc_train_loop(job_data)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    # job_data的数据由运行程序时所传递的参数+hydra/laucher和hydra/local中的参数共同决定
    # 运行程序时已经生成到了/data/ML_document/r3m/evaluation/outputs/BC_pretrained_rep/2023-07-30_04-50-08/job_config.json
    configure_jobs()