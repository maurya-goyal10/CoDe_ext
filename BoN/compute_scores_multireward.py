import os
import json
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

_SCORERS = ['multireward'] # ['strokegen', 'facedetector', 'styletransfer'] 'strokegen', 'facedetector'

_MAP_UG = {
    'styletransfer': {
        'og_img_0': 'style_0',
        'og_img_1': 'style_1',
        'og_img_2': 'style_2',
    },
    'facedetector': {
        'og_img_0': 'og_img_4',
        'og_img_1': 'og_img_6',
        'og_img_2': 'og_img_8',
    }
}

def main():

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext')
                    # else Path('/home/sayak/Projects/PhD_GuidedDiff')
    outputs_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/outputs') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/BoN/outputs')
                    # else Path('/home/sayak/Projects/PhD_GuidedDiff/BoN/outputs_img_abla')

    
    # Load unconditional rewards
    # breakpoint()
    uncond_rewards = dict()
    # _SCORERS = ['pickscore'] # ['strokegen', 'facedetector', 'styletransfer'] 'strokegen', 'facedetector'
    for scorer in _SCORERS:

        scorer_path = outputs_path.joinpath(f'uncond_{scorer}')

        uncond_rewards[scorer] = dict()

        target_dirs = [x for x in scorer_path.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:
            uncond_rewards[scorer][target_dir.stem] = dict()

            # prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            prompt_dirs = [x for x in target_dir.iterdir() if Path.is_dir(x)]

            for prompt_dir in prompt_dirs:

                with open(prompt_dir.joinpath("rewards.json"), 'r') as fp:
                    prompt_reward = json.load(fp)

                uncond_rewards[scorer][target_dir.stem][prompt_dir.stem] = np.array(prompt_reward)
                print(f"{prompt_dir.stem} the rewards are {prompt_reward}")

    # Compute scores
    perf = dict()

    # name_file = 'perf_ccode_b1'
    name_file = 'ablations/multireward_uncond'
    if Path.exists(Path(f'{name_file}.json')):
        with open(f'{name_file}.json', 'r') as fp:
            perf = json.load(fp)
          
    # d = [
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic0_pickscore1_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore0_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore10_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore15_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore20_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore25_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore2_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore3_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore5_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore30_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore50_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore70_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore100_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore150_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore200_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore250_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore300_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore350_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore400_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore450_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore500_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore750_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore1000_multireward_gs2',
    # ]
    
    # d = [
    #     'code4_b5_aesthetic0_pickscore1_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore0_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore10_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore15_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore20_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore25_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore2_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore3_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore5_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore30_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore50_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore70_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore100_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore150_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore200_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore250_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore300_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore350_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore400_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore450_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore500_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore750_multireward',
    #     'code4_greedy_b5_aesthetic1_pickscore1000_multireward',
    # ]
    
    d = ['uncond2_multireward']
    
    # d = [
    #     'FreeDoM_multireward_rho2_aes1_pic0_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic2_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic3_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic5_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic10_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic15_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic20_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic30_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic50_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic70_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic100_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic150_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic200_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic250_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic300_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic350_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic400_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic450_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic500_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic750_multireward',
    #     'FreeDoM_multireward_rho2_aes1_pic1000_multireward',
    # ]

    source_dirs = [x for x in outputs_path.iterdir() if Path.is_dir(x) and x.stem in d]
    # breakpoint()
    
    # print(source_dirs)

    print(perf)

    # breakpoint()
    for source_dir in tqdm(source_dirs):

        # print(source_dir.stem)

        if (source_dir.stem in perf.keys()):
            print('found in perf')
            continue

        # if ('uncond' in source_dir.stem):
        #     print('uncond')
        #     continue

        # if 'uncond2' not in source_dir.stem:
        #     continue

        scorer = source_dir.stem.split('_')[-1] if 'gs' not in source_dir.stem else source_dir.stem.split('_')[-2]
        # scorer = _SCORERS[-1] if 'style' == source_dir.stem.split('_')[1] else _SCORERS[0] # source_dir.stem.split('_')[-1]

        # scorer = source_dir.stem.split('_')[-1]

        if scorer not in _SCORERS:
            continue

        print(source_dir.stem)

        exp_rew = []
        exp_rew_1 = []
        exp_rew_2 = []
        # time_taken_per_sample = []
        win_rate = []
        fids = []
        cmmds = []
        # ref_fids = []
        # ref_cmmds = []

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            if 'test' in source_dir.stem:
                target_key = _MAP_UG[scorer][target_dir.stem]
            else:
                target_key = target_dir.stem

            # prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            prompt_dirs = [x for x in target_dir.iterdir() if Path.is_dir(x)]
            for prompt_dir in prompt_dirs:

                if prompt_dir.stem not in uncond_rewards[scorer][target_key].keys():
                    continue
                    
                # if len([x for x in prompt_dir.iterdir() if (Path.is_file(x) and x.suffix == '.png')]) < len(uncond_rewards[scorer][target_dir.stem][prompt_dir.stem]):
                #     print('less files')
                #     #continue

                with open(prompt_dir.joinpath("rewards.json"), 'r') as fp:
                    prompt_reward = json.load(fp)
                    
                with open(prompt_dir.joinpath("rewards1.json"), 'r') as fp:
                    prompt_reward_1 = json.load(fp)
                    
                with open(prompt_dir.joinpath("rewards2.json"), 'r') as fp:
                    prompt_reward_2 = json.load(fp)
                    
                # with open(prompt_dir.joinpath("time.json"),'r') as f:
                #     time_taken = json.load(f)
                    
                print(f"{prompt_dir.stem} rewards are {prompt_reward}")

                if len(prompt_reward) > len(uncond_rewards[scorer][target_key][prompt_dir.stem]):
                    prompt_reward = prompt_reward[:len(uncond_rewards[scorer][target_key][prompt_dir.stem])]
                if len(prompt_reward_1) > len(uncond_rewards[scorer][target_key][prompt_dir.stem]):
                    prompt_reward_1 = prompt_reward_1[:len(uncond_rewards[scorer][target_key][prompt_dir.stem])]
                if len(prompt_reward_2) > len(uncond_rewards[scorer][target_key][prompt_dir.stem]):
                    prompt_reward_2 = prompt_reward_2[:len(uncond_rewards[scorer][target_key][prompt_dir.stem])]

                exp_rew.append(sum(prompt_reward)/len(prompt_reward))
                exp_rew_1.append(sum(prompt_reward_1)/len(prompt_reward_1))
                exp_rew_2.append(sum(prompt_reward_2)/len(prompt_reward_2))
                win_rate.append((np.array(prompt_reward) > uncond_rewards[scorer][target_key][prompt_dir.stem][:len(prompt_reward)]).astype(int).sum() / len(prompt_reward))
                # time_taken_per_sample.append(time_taken['time_taken']/time_taken['num_images'])

                try:
                    # uncond_path_p = outputs_path.joinpath(f'uncond_{scorer}').joinpath(target_key).joinpath(f'images/{prompt_dir.stem}')
                    uncond_path_p = outputs_path.joinpath(f'uncond2_{scorer}').joinpath(target_key).joinpath(f'{prompt_dir.stem}')
                    
                    # if 'rho' in source_dir.stem:
                    #     if 'mpgd' in source_dir.stem:
                    #         uncond_path_p = outputs_path.joinpath(f'mpgd_ddim100_tt1_rho0_reward_{scorer}').joinpath(target_key).joinpath(f'{prompt_dir.stem}')
                    #         print(f"Fixed for {source_dir.stem}")
                    #     elif 'FreeDoM' in source_dir.stem:
                    #         uncond_path_p = outputs_path.joinpath(f'FreeDoM_aesthetic_rho0_{scorer}').joinpath(target_key).joinpath(f'{prompt_dir.stem}')
                    #         print(f"Fixed for {source_dir.stem}")

                    out = os.popen(f"python {root_path}/pytorch-fid/src/pytorch_fid/fid_score.py '{uncond_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                    fids.append(float(out.split('  ')[-1].split('\n')[0]))

                    out = os.popen(f"python {root_path}/cmmd-pytorch/main.py '{uncond_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                    cmmds.append(float(out.split('  ')[-1].split('\n')[0]))
                except:
                    print(f'Base divergence {prompt_dir.as_posix()}')
                    continue

                # try:
                #     ref_path_p = ref_dirs.joinpath(f'{scorer}').joinpath(target_dir.stem)
                #     out = os.popen(f"python /tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/pytorch-fid/src/pytorch_fid/fid_score.py '{ref_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                #     ref_fids.append(float(out.split('  ')[-1].split('\n')[0]))

                #     out = os.popen(f"python /tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/cmmd-pytorch/main.py '{ref_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                #     ref_cmmds.append(float(out.split('  ')[-1].split('\n')[0]))

                # except:
                #     print(f'Reference divergence {prompt_dir.as_posix()}')
                #     continue

        if len(fids) == 0 or len(cmmds) == 0 or len(exp_rew) == 0 or len(win_rate) == 0:
            continue

        if source_dir.stem not in perf.keys():
            perf[source_dir.stem] = dict()

        perf[source_dir.stem]['exp_rew'] = sum(exp_rew)/len(exp_rew)
        perf[source_dir.stem]['exp_rew_1'] = sum(exp_rew_1)/len(exp_rew_1)
        perf[source_dir.stem]['exp_rew_2'] = sum(exp_rew_2)/len(exp_rew_2)
        perf[source_dir.stem]['win_rate'] = sum(win_rate)/len(win_rate)
        perf[source_dir.stem]['fid'] = sum(fids)/len(fids)
        perf[source_dir.stem]['cmmd'] = sum(cmmds)/len(cmmds)
        # perf[source_dir.stem]['time_taken_per_sample'] = sum(time_taken_per_sample)/(60.0*len(time_taken_per_sample))
        # perf[source_dir.stem]['ref_fid'] = sum(ref_fids)/len(ref_fids)
        # perf[source_dir.stem]['ref_cmmd'] = sum(ref_cmmds)/len(ref_cmmds)
        
        with open(f'{name_file}.json', 'w') as fp:
            json.dump(perf, fp, indent=4)


def ref_divs():

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext')
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff')
    outputs_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/outputs') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/BoN/outputs')
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN/outputs')

    # Compute scores
    perf = dict()

    if Path.exists(Path('perf_cug_ref_divs.json')):
        with open('perf_cug_ref_divs.json', 'r') as fp:
            perf = json.load(fp)

    source_dirs = [x for x in outputs_path.iterdir() if (Path.is_dir(x) and x.stem not in ['plots', 'targets']\
                    and 'test' in x.stem and 'i2i' in x.stem)]
    ref_dirs = "/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/Universal-Guided-Diffusion/stable-diffusion-guided/outputs/targets" if "housky" in currhost\
                else "/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/Universal-Guided-Diffusion/stable-diffusion-guided/outputs/targets"
    # outputs_path.joinpath('targets')
    # breakpoint()
    for source_dir in tqdm(source_dirs):

        # if source_dir.stem in perf.keys():
        #     continue

        scorer = source_dir.stem.split('_')[-1]

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
                    ref_path_p = ref_dirs.joinpath(f'{scorer}').joinpath(target_dir.stem)
                    out = os.popen(f"python {root_path}/pytorch-fid/src/pytorch_fid/fid_score.py '{ref_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                    ref_fids.append(float(out.split('  ')[-1].split('\n')[0]))

                    out = os.popen(f"python {root_path}/cmmd-pytorch/main.py '{ref_path_p.as_posix()}' '{prompt_dir.as_posix()}'").read()
                    ref_cmmds.append(float(out.split('  ')[-1].split('\n')[0]))

                except:
                    print(f'Reference divergence {prompt_dir.as_posix()}')
                    continue

        if len(ref_cmmds) == 0 or len(ref_fids) == 0:
            continue

        if source_dir.stem not in perf.keys():
            perf[source_dir.stem] = dict()

        perf[source_dir.stem]['ref_fid'] = sum(ref_fids)/len(ref_fids)
        perf[source_dir.stem]['ref_cmmd'] = sum(ref_cmmds)/len(ref_cmmds)
        
        with open('perf_cug_ref_divs.json', 'w') as fp:
            json.dump(perf, fp)

def create_folders():

    select_samples = 5

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext')
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff')
    outputs_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/outputs') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/BoN/outputs')
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN/outputs')

    newpath = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/cherry_picking') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/BoN/cherry_picking2')
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN/cherry_picking2')
    
    source_dirs = [x for x in outputs_path.iterdir() if (Path.is_dir(x) and x.stem != 'plots')]
    for source_dir in tqdm(source_dirs):

        if 'code_grad' not in source_dir.stem or 'b1_' in source_dir.stem:
            continue

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            if not Path.exists(newpath.joinpath(source_dir.stem).joinpath(target_dir.stem)):
                Path.mkdir(newpath.joinpath(source_dir.stem).joinpath(target_dir.stem), exist_ok=True, parents=True)
            
            image = Image.open(target_dir.joinpath('images/target.png'))
            image.save(newpath.joinpath(source_dir.stem).joinpath(target_dir.stem).joinpath(f'target.png'))

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            for prompt_dir in prompt_dirs:
                
                folder = newpath.joinpath(source_dir.stem).joinpath(target_dir.stem).joinpath('images').joinpath(prompt_dir.stem)
                if not Path.exists(folder):
                    Path.mkdir(folder, exist_ok=True, parents=True)
                    # print(folder.as_posix())

                num_samples = len([x for x in prompt_dir.iterdir() if x.suffix == '.png'])
                if select_samples < num_samples:
                    select_samples = num_samples

                num_selected = len([x for x in folder.iterdir() if x.suffix == '.png'])
                if num_selected == select_samples:
                    print('selection done!!')
                    continue

                with open(prompt_dir.joinpath('rewards.json'), 'r') as fp:
                    rewards = json.load(fp)

                selected_indexes = [x for x in np.argsort(rewards)[::-1][:5]]

                for idx in selected_indexes:
                    image = Image.open(prompt_dir.joinpath(f'{idx}.png'))
                    image.save(folder.joinpath(f'{idx}.png'))

        #         break

        #     break

        # break

if __name__ == '__main__':
    # ref_divs()
    # target_divs()
    main()
    # create_folders()