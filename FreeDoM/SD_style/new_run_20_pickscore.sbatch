#!/bin/sh

# ------------------------------------------------------------------------------
# Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --output=new_run_20_pick.out
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

python txt2img.py --prompt "A small bathroom with a tub, toilet, sink, and a laundry basket are shown." "A bus stopped on the side of the road while people board it." "A bunch of people posing with some bikes." "a jet engine on the wing of a plane" "A bunch of bicycles parked on the street with items sitting around them" "A dog standing in front of a doorway." --ddim_steps 100 --n_iter 10 --seed 2025 --H 512 --W 512 --rho 0

echo -n "This run completed on: "
date