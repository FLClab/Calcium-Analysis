#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --account=def-flavielc
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x-%A_%a.out


#### PARAMETERS
# Use this directory venv, reusable across RUNs
module load python/3.8 scipy-stack/2022a
module load cuda cudnn
VENV_DIR=${HOME}/venvs/unet-bcd
source $VENV_DIR/bin/activate

# Moves to working directory
cd ${HOME}/Documents/mscts-analysis/notebooks

config="./configs/config-unet3d_default.yml"
seed=42

# Launch training 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Started training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo Config: $config
echo Seed: $seed

python train.py --config $config --seed $seed

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "% Done training"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"