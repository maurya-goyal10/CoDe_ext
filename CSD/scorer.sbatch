#!/bin/sh

# ------------------------------------------------------------------------------
# Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --partition=ptxd-a100
#SBATCH --time=24:00:00
#SBATCH --output=scorer.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=mv_res


# -----------------------------------------------------------------------------
# Printing some information
# -----------------------------------------------------------------------------

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
eval "$(/glb/data/ptxd_dash/nlasqh/anaconda3/bin/conda shell.bash hook)"

conda activate ldm3

export PYTHONDONTWRITEBYTECODE=abc
export PYTHONUNBUFFERED=TRUE
PROXY=zproxy-global.shell.com:80
export NO_PROXY=.shell.com
export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY

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