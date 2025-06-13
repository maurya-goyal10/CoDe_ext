#!/bin/sh

# ------------------------------------------------------------------------------
# Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --output=multireward_30.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --gres=gpu:a40

# ------------------------------------------------------------------------------
# Printing some information
# -----------------------------------------------------------------------------

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module load miniconda/3.9

conda activate mpgd

export PYTHONDONTWRITEBYTECODE=abc
export PYTHONUNBUFFERED=TRUE

# ------------------------------------------------------------------------------
# And finally running the code
# ------------------------------------------------------------------------------

echo "--------------- Running the code ---------------"

echo -n "This run started on: "
date

python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi

# cd ../../src_sd

python txt2img.py --prompt "snail" "hippopotamus" "cheetah" "crocodile" "lobster" "octopus" --ddim_steps 100 --n_iter 10 --seed 2024 --H 512 --W 512 --rho 0.2 --weight1 1 --weight2 30

echo -n "This run completed on: "
date