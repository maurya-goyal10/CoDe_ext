import os
import re

from pathlib import Path

def create_job_file(config, export_path):
    """Create the slurm job file
    """

    currhost = os.uname()[1]
    template_path = Path('pytorch_shell.job') if "housky" in currhost else Path('pytorch.job') # Template file

    lines = []

    with open(template_path, 'r') as fp:
        for line in fp:
            lines.append(line)

    updated_lines = []

    for line in lines:
        if re.search("output", line):
            updated_lines.append(line.replace("slurm", config))

        elif re.search("--config", line):
            
            temp = line

            temp = temp.replace("method", export_path.stem)
            temp = temp.replace("config_file", config)
            updated_lines.append(temp)

        else:
            updated_lines.append(line)

    with open(export_path.joinpath(f'{config}.sbatch'), 'w') as fp:
        for line in updated_lines:
            fp.write(line)

    return


def main():
    """Generate job files for available configurations
    """

    config_dir = Path('../configs_split')

    methods = [x for x in config_dir.iterdir() if x.is_dir()]

    for method in methods:

        if method.stem not in ['code_grad_final_general']: # 'code_grad', 'uncond'
            continue

        curr_path = Path(method.stem)
        # print(curr_path)

        if not Path.exists(curr_path):
            Path.mkdir(curr_path, parents=True, exist_ok=True)

        configs = [x for x in method.iterdir() if x.is_file()]
        # breakpoint()
        for config in configs: 
            
            # if 'p0' in config.stem:
            #     continue
            if 'pickscore' in config.stem:
                continue
            # if not 'general4_' in config.stem:
            #     continue
            # if not any(clustering in config.stem for clustering in ('KMeans','HDBSCAN')):
            #     continue
            # if('code40' not in config.stem):
            #     continue
            # if ('var4revi' not in config.stem):
            #     continue
            # if('_st7_et3_' not in config.stem):
            #     continue
            # if not any(temp in config.stem for temp in ('temp2000_', 'temp3000_', 'temp4000_', 'temp6000_')):
            #     continue
            # if ('temp1000_' not in config.stem and '5000' not in config.stem):
            #     continue
            # if  ('pickscore' not in config.stem) or ('st7' not in config.stem) or ('et3' not in config.stem) or ('gs2' not in config.stem):
            #     continue
            # if ('compress' in config.stem) or ('stroke' in config.stem):
            #     continue
            # if 'code1_' not in config.stem and 'gs3' not in config.stem:
            #     continue
            # if 'code40' in config.stem:
            print(config.stem)
            create_job_file(config.stem, curr_path)

if __name__ == '__main__':
    main()