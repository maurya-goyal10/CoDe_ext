#!/bin/sh

# ------------------------------------------------------------------------------
# Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --output=slurm_pickscore_60.out
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

python style.py --ddim_steps 100 --n_iter 10 --H 512 --W 512 --scale 5.0 --rho 0 --tt 1 --prompt "a bus with a view of a lot of traffic and the back of another bus with a billboard on the back end" "A young man riding through the air on top of a skateboard." "A toy elephant is sitting inside a wooden car toy." "A motorized bicycle covered with greens and beans." "A man sitting at a table in front of bowls of spices." "there is a bathroom that has a lot of things on the floor" --seed 2025 --reward_model "PickScore" --start_ratio 0.7 --end_ratio 0.3

echo -n "This run completed on: "
date