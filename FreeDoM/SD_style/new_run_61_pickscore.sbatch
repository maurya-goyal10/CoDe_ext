#!/bin/sh

# ------------------------------------------------------------------------------
# Slurm directives
# -----------------------------------------------------------------------------

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --output=new_run_61_pickscore.out
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

python txt2img.py --prompt "A large commercial airliner silhoetted in the sun." "Residential bathroom with modern design and tile floor." "a bus with a view of a lot of traffic and the back of another bus with a billboard on the back end" "A young man riding through the air on top of a skateboard." "A toy elephant is sitting inside a wooden car toy." "A motorized bicycle covered with greens and beans." --ddim_steps 100 --n_iter 10 --seed 2024 --H 512 --W 512 --rho 0.2

echo -n "This run completed on: "
date