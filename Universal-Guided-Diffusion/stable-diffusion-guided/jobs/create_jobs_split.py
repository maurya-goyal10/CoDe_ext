#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-Aug-2024
# 
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import re, os
import numpy as np

from pathlib import Path

_TASKS = {
    # 'aesthetic': {
    #     'weights': [30],
    #     'cmd': 'scripts/aesthetic.py --scale 5.0 --optim_forward_guidance --optim_num_steps 6 --optim_original_conditioning --ddim_steps 100'
    # },
    'pickscore': {
        'weights': [0],
        'cmd': 'scripts/pickscore.py --scale 5.0 --optim_forward_guidance --optim_num_steps 1 --optim_original_conditioning --ddim_steps 100'
    },
    # 'face': {
    #     'weights': [20000], #[5000, 10000, 20000, 30000, 40000],
    #     'cmd': 'scripts/face_detection.py --optim_forward_guidance --fr_crop --optim_num_steps 2 --optim_original_conditioning --ddim_steps 500'
    # },
    # 'style': {
    #     'weights': [1, 3, 6, 12, 24],
    #     'cmd': 'scripts/style_transfer.py --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_original_conditioning --ddim_steps 500'
    # },
    # 'face_i2i': {
    #     'weights': [5000, 10000, 20000, 30000, 40000],
    #     'cmd': 'scripts/face_detection_i2i.py --optim_forward_guidance --fr_crop --optim_num_steps 2 --optim_original_conditioning --ddim_steps 500'
    # },
    # 'style_i2i': {
    #     'weights': [1, 3, 6, 12, 24],
    #     'cmd': 'scripts/style_transfer_i2i.py --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_original_conditioning --ddim_steps 500'
    # },
    # 'strokegen': {
    #     'weights': [6],
    #     'cmd': 'scripts/stroke_transfer.py --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_original_conditioning --ddim_steps 500'
    # },
}

currhost = os.uname()[1]
if 'housky' in currhost: # shell cluster
    MODEL_CHECKPOINT = '/glb/data/ptxd_dash/nlasqh/data/models/SD/v1-5-pruned-emaonly.ckpt'
elif currhost == 'tud1006406':  # sayak desktop
    MODEL_CHECKPOINT = '/home/sayak/Projects/PhD_GuidedDiff/Universal-Guided-Diffusion/stable-diffusion-guided/ckpts/v1-5-pruned-emaonly.ckpt'
else: # cluster
    MODEL_CHECKPOINT = '/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/Universal-Guided-Diffusion/stable-diffusion-guided/ckpts/v1-5-pruned-emaonly.ckpt'

def create_job_file(param, export_path, task, target_idx, prompt_idx, strength=None):
    """Create the slurm job file
    """

    if 'housky' in currhost: # shell cluster
        template_path = Path('pytorch_shell.job') # Template file
    else:
        template_path = Path('pytorch.job')

    lines = []

    with open(template_path, 'r') as fp:
        for line in fp:
            lines.append(line)

    updated_lines = []

    for line in lines:
        if re.search("--output=slurm.out", line):
            if strength is None:
                updated_lines.append(line.replace("slurm", f"{task[:3]}_{param}_{target_idx}_{prompt_idx}"))
            else:
                updated_lines.append(line.replace("slurm", f"{task[:3]}_{param}_r{int(float(round(strength,1))*10)}_{target_idx}"))

        elif re.search("commandline", line):
            
            if strength is None:
                if task == 'aesthetic':
                    command = f'{_TASKS[task]["cmd"]} --text_type {prompt_idx} --indexes {target_idx} --optim_forward_guidance_wt {param} --optim_folder ./outputs/test_{task}_{param}/images --ckpt {MODEL_CHECKPOINT} --trials 10 --seed 2024'
                else:
                    command = f'{_TASKS[task]["cmd"]} --prompt_indexes {prompt_idx} --indexes {target_idx} --optim_forward_guidance_wt {param} --optim_folder ./outputs/test_{task}_{param} --ckpt {MODEL_CHECKPOINT} --trials 10 --seed 2024'
            else:
                command = f'{_TASKS[task]["cmd"]} --text_type {prompt_idx} --indexes {target_idx} --optim_forward_guidance_wt {param} --optim_folder ./outputs/test_{task}_{param}_r{int(float(round(strength,1))*10)} --ckpt {MODEL_CHECKPOINT} --strength {float(round(strength,1))} --trials 10'
            
            updated_lines.append(line.replace("commandline", command))

        else:
            updated_lines.append(line)

    if strength is None:
        jobfile = export_path.joinpath(f'{task[:3]}_{param}_{target_idx}_p{prompt_idx}.sbatch')
    else:
        jobfile = export_path.joinpath(f'{task[:3]}_{param}_r{int(float(round(strength,1))*10)}_{target_idx}_p{prompt_idx}.sbatch')
    
    with open(jobfile, 'w') as fp:
        for line in updated_lines:
            fp.write(line)

    return


def main():
    """Generate job files for available configurations
    """

    for task in _TASKS.keys():

        save_path = Path(task).joinpath(Path(f'{task}_split'))

        if not Path.exists(save_path):
            Path.mkdir(save_path, parents=True, exist_ok=True)

        num_targets = 1
        num_prompts = 51  if task == 'aesthetic' else 50
        if task == 'aesthetic' or task=='pickscore':
            num_targets = 1 # no target. generate all samples based on text

        if 'i2i' in task:
            for strength in np.arange(0.4, 1.0, 0.1):
                for target_idx in range(num_targets):
                    for prompt_idx in range(num_prompts):
                        for forward_wt in _TASKS[task]['weights']:
                            create_job_file(param=forward_wt, 
                                            export_path=save_path, 
                                            task=task, 
                                            target_idx=target_idx,
                                            prompt_idx=prompt_idx,
                                            strength=strength)
        else:
            for target_idx in range(num_targets):
                for prompt_idx in range(num_prompts):
                    for forward_wt in _TASKS[task]['weights']:
                        create_job_file(param=forward_wt, 
                                        export_path=save_path, 
                                        task=task, 
                                        target_idx=target_idx,
                                        prompt_idx=prompt_idx)
            

if __name__ == '__main__':
    main()