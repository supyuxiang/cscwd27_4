#!/bin/bash
#SBATCH --job-name=thinkloop
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/yxfeng/cscwd27_thinkloop
mkdir -p logs
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
/home/yxfeng/miniconda3/envs/www27/bin/python collect.py configs/weibo.yaml
/home/yxfeng/miniconda3/envs/www27/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --master_port=29741 \
  trainer.py configs/weibo.yaml
