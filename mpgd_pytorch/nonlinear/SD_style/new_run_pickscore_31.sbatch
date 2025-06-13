#!/bin/sh

# ------------------------------------------------------------------------------
# Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --output=slurm_pickscore_31.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
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

python style.py --ddim_steps 100 --n_iter 10 --H 512 --W 512 --scale 5.0 --rho 7.5 --tt 1 --prompt "Two small planes sitting near each other on a run way." "there is a bus that has a bike attached to the front" "A bird that is sitting in the rim of a tire." "The black motorcycle is parked on the sidewalk." "A corner of a rest room with a big shower." "a dog with a plate of food on the ground"  --seed 2024 --reward_model "PickScore" --start_ratio 0.7 --end_ratio 0.3

echo -n "This run completed on: "
date