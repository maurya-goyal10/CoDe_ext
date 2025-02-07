#!/bin/sh

# ------------------------------------------------------------------------------
# Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --output=compute_clipscores.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000
#SBATCH --gres=gpu:a40:1

# ------------------------------------------------------------------------------
# Printing some information
# -----------------------------------------------------------------------------

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
module use /opt/insy/modulefiles
module load cuda/11.3
module load miniconda/3.9

conda activate ldm

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

python compute_scores.py

echo -n "This run completed on: "
date