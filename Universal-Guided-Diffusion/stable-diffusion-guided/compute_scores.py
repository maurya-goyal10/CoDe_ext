import os
import json
import numpy as np

from tqdm.auto import tqdm
from pathlib import Path

_SCORERS = ['facedetector', 'styletransfer'] # 'facedetector', 'styletransfer', 'strokegen'

# _MAP_UG = {
#     'styletransfer': {
#         'og_img_0': 'style_0',
#         'og_img_1': 'style_1',
#         'og_img_2': 'style_2',
#     },
#     'facedetector': {
#         'og_img_0': 'og_img_4',
#         'og_img_1': 'og_img_6',
#         'og_img_2': 'og_img_8',
#     },
#     'strokegen': {
#         'og_img_0': 'og_img_0',
#         'og_img_1': 'og_img_1',
#         'og_img_2': 'og_img_2',
#     }
# }

def main():

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff')
    outputs_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/Universal-Guided-Diffusion/stable-diffusion-guided/outputs') if "housky" in currhost\
                    else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/Universal-Guided-Diffusion/stable-diffusion-guided/outputs_addons')
    

    # Load unconditional rewards
    uncond_rewards = dict()
    for scorer in _SCORERS:

        scorer_path = outputs_path.joinpath(f'uncond2_{scorer}')

        uncond_rewards[scorer] = dict()

        target_dirs = [x for x in scorer_path.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:
            uncond_rewards[scorer][target_dir.stem] = dict()

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]

            for prompt_dir in prompt_dirs:

                with open(prompt_dir.joinpath("rewards.json"), 'r') as fp:
                    prompt_reward = json.load(fp)

                uncond_rewards[scorer][target_dir.stem][prompt_dir.stem] = np.array(prompt_reward)

    # Compute scores
    perf = dict()

    if Path.exists(Path('perf_addons_ug.json')):
        with open('perf_addons_ug.json', 'r') as fp:
            perf = json.load(fp)

    source_dirs = [x for x in outputs_path.iterdir() \
                    if (Path.is_dir(x) and x.stem != 'plots')]

    for source_dir in tqdm(source_dirs):

        if source_dir.stem in perf.keys():
            continue

        if 'uncond' in source_dir.stem:
            continue

        print(source_dir.stem)

        scorer = _SCORERS[1] if 'style' == source_dir.stem.split('_')[1] else _SCORERS[0] # source_dir.stem.split('_')[-1]

        if scorer not in _SCORERS:
            continue

        exp_rew = []
        win_rate = []
        fids = []
        cmmds = []

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            # target_key = _MAP_UG[scorer][target_dir.stem]
            target_key = target_dir.stem

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            for prompt_dir in prompt_dirs:

                if prompt_dir.stem not in uncond_rewards[scorer][target_key].keys():
                    continue
                    
                if len([x for x in prompt_dir.iterdir() if (Path.is_file(x) and x.suffix == '.png')]) == 0:
                    continue

                with open(prompt_dir.joinpath("rewards.json"), 'r') as fp:
                    prompt_reward = json.load(fp)

                if len(prompt_reward) > len(uncond_rewards[scorer][target_key][prompt_dir.stem]):
                    prompt_reward = prompt_reward[:len(uncond_rewards[scorer][target_key][prompt_dir.stem])]

                exp_rew.append(sum(prompt_reward)/len(prompt_reward))
                win_rate.append((np.array(prompt_reward) > uncond_rewards[scorer][target_key][prompt_dir.stem][:len(prompt_reward)]).astype(int).sum() / len(prompt_reward))

                uncond_path_p = outputs_path.joinpath(f'uncond2_{scorer}').joinpath(target_key).joinpath(f'images/{prompt_dir.stem}')
                out = os.popen(f"python {root_path}/pytorch-fid/src/pytorch_fid/fid_score.py '{uncond_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                fids.append(float(out.split('  ')[-1].split('\n')[0]))

                out = os.popen(f"python {root_path}/cmmd-pytorch/main.py '{uncond_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                cmmds.append(float(out.split('  ')[-1].split('\n')[0]))

        if source_dir.stem not in perf.keys():
            perf[source_dir.stem] = dict()

        perf[source_dir.stem]['exp_rew'] = sum(exp_rew)/len(exp_rew)
        perf[source_dir.stem]['win_rate'] = sum(win_rate)/len(win_rate)
        perf[source_dir.stem]['fid'] = sum(fids)/len(fids)
        perf[source_dir.stem]['cmmd'] = sum(cmmds)/len(cmmds)
        print(perf[source_dir.stem])
        # save_perf = None
        # if Path.exists(Path('perf.json')):
        #     with open('perf.json', 'r') as fp:
        #         save_perf = json.load(fp)
            
        # save_perf[source_dir.stem] = perf[source_dir.stem]
        
        with open('perf_addons_ug.json', 'w') as fp:
            json.dump(perf, fp)

def ref_divs():

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff')
    outputs_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/outputs') if "housky" in currhost\
                    else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/Universal-Guided-Diffusion/stable-diffusion-guided/outputs')

    # Compute scores
    perf = dict()

    if Path.exists(Path('perf_cug_refdivs.json')):
        with open('perf_cug_refdivs.json', 'r') as fp:
            perf = json.load(fp)

    source_dirs = [x for x in outputs_path.iterdir()\
                    if (Path.is_dir(x) and x.stem not in ['plots', 'targets'] and 'test' in x.stem and 'i2i' in x.stem)]
    print(source_dirs)
    ref_dirs = "/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/Universal-Guided-Diffusion/stable-diffusion-guided/outputs/targets"  # outputs_path.joinpath('targets')
    # breakpoint()
    for source_dir in tqdm(source_dirs):

        # if source_dir.stem in perf.keys():
        #     continue

        scorer = _SCORERS[1] if 'style' == source_dir.stem.split('_')[1] else _SCORERS[0] # scorer = source_dir.stem.split('_')[-1]

        if scorer not in _SCORERS:
            continue

        ref_fids = []
        ref_cmmds = []

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            for prompt_dir in prompt_dirs:
                    
                # if len([x for x in prompt_dir.iterdir() if (Path.is_file(x) and x.suffix == '.png')]) < len(uncond_rewards[scorer][target_dir.stem][prompt_dir.stem]):
                #     print('less files')
                #     continue
                try:
                    ref_path_p = Path(ref_dirs).joinpath(f'{scorer}').joinpath(f'{target_dir.stem}')
                    out = os.popen(f"python {root_path}/pytorch-fid/src/pytorch_fid/fid_score.py '{ref_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                    ref_fids.append(float(out.split('  ')[-1].split('\n')[0]))

                    out = os.popen(f"python {root_path}/cmmd-pytorch/main.py '{ref_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                    ref_cmmds.append(float(out.split('  ')[-1].split('\n')[0]))

                except Exception as e:
                    print(f'Reference divergence {prompt_dir.as_posix()}')
                    print(target_dir.stem)
                    print(e)
                    continue

        if len(ref_cmmds) == 0 or len(ref_fids) == 0:
            continue

        if source_dir.stem not in perf.keys():
            perf[source_dir.stem] = dict()

        perf[source_dir.stem]['ref_fid'] = sum(ref_fids)/len(ref_fids)
        perf[source_dir.stem]['ref_cmmd'] = sum(ref_cmmds)/len(ref_cmmds)
        
        with open('perf_cug_refdivs.json', 'w') as fp:
            json.dump(perf, fp)



if __name__ == '__main__':
    # main()
    ref_divs()