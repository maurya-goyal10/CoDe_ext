import os
import json
import numpy as np
# import seaborn as sns
import pandas as pd
# import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm

_SCORERS = ['multireward'] # ['compress'] # ['strokegen', 'facedetector', 'styletransfer'] 

def compute_clipscore():

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext')
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff')
    base_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN') if "housky" in currhost\
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/BoN')
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN')
                    

    perf = dict()

    filename = 'ablations/multireward_code4_clip.json'
    if Path.exists(base_path.joinpath(filename)):
        with open(base_path.joinpath(filename), 'r') as fp:
            perf = json.load(fp)

    # Load unconditional rewards
    uncond_clipscores = dict()
    for scorer in _SCORERS:

        scorer_path = base_path.joinpath('outputs').joinpath(f'uncond_{scorer}')

        uncond_clipscores[scorer] = dict()

        clip_scores = []

        target_dirs = [x for x in scorer_path.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:
            uncond_clipscores[scorer][target_dir.stem] = dict()

            # prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            prompt_dirs = [x for x in target_dir.iterdir() if Path.is_dir(x)]

            for prompt_dir in prompt_dirs:

                if not Path.exists(prompt_dir.joinpath('clipscores.json')):

                    # json_path = target_dir.joinpath("images").joinpath(f"{prompt_dir.stem}.json")
                    json_path = target_dir.joinpath(f"{prompt_dir.stem}.json")
                    captions = {x.stem: prompt_dir.stem for x in prompt_dir.iterdir() if x.suffix == '.png'}
                    with open(json_path, 'w') as fp:
                        json.dump(captions, fp)
                    
                    try:
                        store_path = prompt_dir.joinpath('clipscores.json')
                        score = os.popen(f"python clipscore.py '{json_path.as_posix()}' '{prompt_dir.as_posix()}' --save_per_instance '{store_path.as_posix()}'").read()
                        clip_scores.append(float(score.split(': ')[1]))
                    except:
                        print(prompt_dir)
                        print(json_path)
                        raise ValueError()
                    
                with open(prompt_dir.joinpath("clipscores.json"), 'r') as fp:
                    prompt_clipscores = json.load(fp)

                image_ids = [x.stem for x in prompt_dir.iterdir() if x.suffix == '.png']
                uncond_clipscores[scorer][target_dir.stem][prompt_dir.stem] = np.array([prompt_clipscores[image_id]['CLIPScore'] for image_id in image_ids])
                clip_scores = uncond_clipscores[scorer][target_dir.stem][prompt_dir.stem].tolist()

        if scorer_path.stem not in perf.keys():
            perf[scorer_path.stem] = dict()
            perf[scorer_path.stem]['clipscore'] = sum(clip_scores)/len(clip_scores)
        
        perf[scorer_path.stem]['clipwinrate'] = 0.5
        
    # source_dirs = [x for x in base_path.joinpath('outputs').iterdir()\
    #                 if (Path.is_dir(x) and x.stem != 'plots' and 'c_code_' in x.stem and 'compress' in x.stem)]
    
    
    # d = ['code_grad4_b5_aesthetic_gs0',
    #      'code_grad4_b5_aesthetic_gs3',
    #      'code_grad4_b5_aesthetic_gs5',
    #      'code40_b5_aesthetic']
    
    # d = ['code40_greedy_b5_aesthetic',
    #      'uncond2_aesthetic',
    #      'code4_greedy_b5_aesthetic',
    #      'code_grad_final_general4_greedy_b5_gb5_st6_et2_FreeDoM_aesthetic_gs3',
    #      'code_grad_final_general4_greedy_b5_gb5_st6_et1_FreeDoM_aesthetic_gs2',
    #      'code_grad_final_general4_greedy_b5_gb5_st6_et1_FreeDoM_aesthetic_gs3',
    #      'code_grad_final_general4_greedy_b5_gb5_st7_et1_FreeDoM_aesthetic_gs2',
    #      'code_grad_final_general4_greedy_b5_gb5_st7_et1_FreeDoM_aesthetic_gs3',
    #      'code_grad_final_general4_greedy_b5_gb5_st7_et3_FreeDoM_aesthetic_gs3',
    #      'code_grad_final_general4_greedy_b5_gb5_st6_et3_FreeDoM_aesthetic_gs3',
    #      'code_grad_final_general4_greedy_b5_gb5_st7_et2_FreeDoM_aesthetic_gs2',
    #      'code_grad_final_general4_greedy_b5_gb5_st8_et3_FreeDoM_aesthetic_gs3',
    #      'code_grad_final_general4_greedy_b5_gb5_st8_et2_FreeDoM_aesthetic_gs2',
    #      'code_grad_final_general4_greedy_b5_gb5_st8_et3_FreeDoM_aesthetic_gs2',
    #      'code_grad_final_general4_greedy_b5_gb5_st8_et2_FreeDoM_aesthetic_gs3',
    #      'code_grad_final_general4_greedy_b5_gb5_st7_et2_FreeDoM_aesthetic_gs3',
    #      'code4_multinomial_temp1000_b5_aesthetic'
    #     ]
    
    # d = ['code4_greedy_b5_aesthetic', 
    #      'codevar4_greedy_b5_aesthetic',
    #      'codevar4i_greedy_b5_aesthetic',
    #      'codevar4ii_greedy_b5_aesthetic',
    #      'codevar4iii_greedy_b5_aesthetic',
    #      'codevar4iiii_greedy_b5_aesthetic',
    #      'codevar4iiiii_greedy_b5_aesthetic',
    #      'codevar4rev_greedy_b5_aesthetic',
    #      'codevar4revi_greedy_b5_aesthetic',
    #      'codevar4revii_greedy_b5_aesthetic',
    #      'codevar4reviii_greedy_b5_aesthetic',
    #      'codevar4reviiii_greedy_b5_aesthetic',
    #      'codevar4reviiiii_greedy_b5_aesthetic',
    #      'codevar4reviiiiii_greedy_b5_aesthetic',
    #      'codevar4reviiiiiii_greedy_b5_aesthetic',
    #     ]
    # d = ['code40_greedy_b5_aesthetic',
    #     'uncond2_aesthetic',
    #     'code4_greedy_b5_aesthetic',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et2_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et1_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et1_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et1_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et1_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et3_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et3_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et2_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et3_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et2_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et3_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et2_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et2_FreeDoM_aesthetic_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et1_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st9_et0_FreeDoM_aesthetic_gs15',
    #     'code4_multinomial_temp1000_b5_aesthetic'
    # ]
    # d = [
    #     'code40_greedy_b5_pickscore',
    #     'code4_greedy_b5_pickscore',
    #     'uncond2_pickscore',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st9_et0_FreeDoM_pickscore_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st9_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et0_FreeDoM_pickscore_gs3',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et0_FreeDoM_pickscore_gs2',
    # ]
    # d = ['code40_b5_aesthetic',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4i_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4ii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4iii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4iiii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4iiiii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4rev_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4revi_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4revii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4reviii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4reviiii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4reviiiii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4reviiiiii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_generalvar4reviiiiiii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2']
    # d = ['code4_b5_aesthetic',
    #     'code4_multinomial_temp200_b5_aesthetic',
    #     'code4_multinomial_temp500_b5_aesthetic',
    #     'code4_multinomial_temp750_b5_aesthetic',
    #     'code4_multinomial_temp1000_b5_aesthetic',
    #     'code4_multinomial_temp2000_b5_aesthetic',
    #     'code4_multinomial_temp3000_b5_aesthetic',
    #     'code4_multinomial_temp4000_b5_aesthetic',
    #     'code4_multinomial_temp5000_b5_aesthetic',
    #     'code4_multinomial_temp6000_b5_aesthetic',
    #     'code4_multinomial_temp7000_b5_aesthetic',
    #     'code4_multinomial_temp8000_b5_aesthetic',
    #     'code4_multinomial_temp9000_b5_aesthetic',
    #     'code4_multinomial_temp10000_b5_aesthetic',
    #     'code4_multinomial_temp15000_b5_aesthetic',]
    # d = ['code_grad_final_general4_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et0_HDBSCAN_FreeDoM_aesthetic_gs20',
    #     'code_grad_final_general4_greedy_b5_gb5_st6_et0_KMeans_FreeDoM_aesthetic_gs20']
                
    # d = ['code_grad_final_general4_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    #      'code_grad_final_general4_multinomial_b5_gb5_temp1000_st6_et0_FreeDoM_aesthetic_gs2',
    #      "code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp1000_st6_et0_FreeDoM_aesthetic_gs2"]
    # d = [
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp100_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp200_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp350_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp500_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp750_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp1000_st6_et0_FreeDoM_aesthetic_gs2',
    #     ]
    
    # table_1_aesthetic_clip
    # d = [
    # 'uncond_aesthetic',
    # 'code4_greedy_b5_aesthetic',
    # 'code40_greedy_b5_aesthetic',
    # 'code_grad_final_general4_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_general4_multinomial_b5_gb5_temp1000_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_greedy_b5_gb5_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp1000_st6_et0_FreeDoM_aesthetic_gs2',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp1000_st6_et0_KMeans_FreeDoM_aesthetic_gs20'
    #     ]
    # d = [
    #     'uncond2_pickscore',
    #     'code4_greedy_b5_pickscore',
    #     'code40_greedy_b5_pickscore',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiiiiii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2'
    # ]
    # d = [
    # 'mpgd_ddim100_tt1_rho75_reward_aesthetic',
    # 'FreeDoM_aesthetic_rho2_aesthetic',
    # 'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp1000_st6_et0_FreeDoM_aesthetic_gs2'
    # ]
    
    # d = ['code40_b5_pickscore',
    #     'code_grad_final_general4_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4i_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4ii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4iii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4iiii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4iiiii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4rev_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4revi_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4revii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiiii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiiiii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiiiiii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4new_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newii_greedy_b5_gb5_st7_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newii_greedy_b5_gb5_st95_et0_FreeDoM_pickscore_gs2'
    #     ]
    
    # d = ['code40_greedy_b5_pickscore',
    #     'code4_greedy_b5_pickscore',
    #     'uncond2_pickscore',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st9_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st8_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st95_et0_FreeDoM_pickscore_gs2',
    # ]
    
    #ablations_compress_testing_
    # d = [
    #     # 'code4_greedy_b5_compress',
    #     # 'code40_greedy_b5_compress',
    #     # 'code_grad_final_general4_greedy_b5_gb5_st10_et0_antithetic_1_FreeDoM_compress_gs2',
    #     # 'code_grad_final_general4_greedy_b5_gb5_st10_et0_antithetic_5_FreeDoM_compress_gs2',
    #     # 'code_grad_final_general4_greedy_b5_gb5_st10_et0_antithetic_10_FreeDoM_compress_gs2',
    #     # 'code_grad_final_general4_greedy_b5_gb5_st10_et0_antithetic_50_FreeDoM_compress_gs2',
    # ]
    
    # d = ['code40_b5_pickscore',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4i_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4ii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4iii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4iiii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4iiiii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4rev_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4revi_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4revii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiiii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiiiii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4reviiiiiii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4new_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newiii_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     ]
    
    # ablation_pickscore_temp_newi_clip_
    # d = [
    #     'code_grad_final_generalvar4newi_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp500_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp1000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp2000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp3000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp4000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp5000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp7000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp10000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp12000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp15000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp16000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp18000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp20000_st10_et0_FreeDoM_pickscore_gs2',
    # ]
    
    # table_2_aesthetic_clip
    # d = [
    #     'code_grad_final_generalvar4reviiiiii_multinomial_b5_gb5_temp1000_st6_et0_FreeDoM_aesthetic_gs2',
    #     'mpgd_ddim100_tt1_rho75_reward_aesthetic',
    #     'FreeDoM_aesthetic_rho2_aesthetic',
    #     'ug_aesthetic_rho_30_cfg_50_w_6_aesthetic',
    # ]
    
    # table_1_pickscore
    # d = [
    #     # 'code40_greedy_b5_pickscore',
    #     # 'code4_greedy_b5_pickscore',
    #     'uncond2_pickscore',
    #     # 'code_grad_final_general4_multinomial_b5_gb5_temp3000_st10_et0_FreeDoM_pickscore_gs2',
    #     # 'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     # 'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp3000_st10_et0_FreeDoM_pickscore_gs2',
    #     # 'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp3000_st10_et0_KMeans_FreeDoM_pickscore_gs20',
    #     # 'code_grad_final_generalvar4newi_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    # ]
    
    # multireward
    # d = [
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic0_pickscore1_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore0_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore10_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore15_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore20_multireward_gs2',
    #     'code_grad_final_general4_greedy_b5_gb5_st10_et0_FreeDoM_aesthetic1_pickscore25_multireward_gs2',
    # ]
    
    #table_2_pickscore
    # d = [
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp3000_st10_et0_FreeDoM_pickscore_gs2',
    #     'mpgd_ddim100_tt1_rho75_reward_pickscore',
    #     'FreeDoM_pickscore_rho15_pickscore'
    # ]
    
    # ablation_pickscore_temp_newi__
    # d = [
    #     'code_grad_final_generalvar4newi_greedy_b5_gb5_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp3000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp25000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp30000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp40000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp50000_st10_et0_FreeDoM_pickscore_gs2',
    #     'code_grad_final_generalvar4newi_multinomial_b5_gb5_temp100000_st10_et0_FreeDoM_pickscore_gs2',
    # ]

    # multireward_code4_clip
    d = [
        'code4_b5_aesthetic0_pickscore1_multireward',
        'code4_greedy_b5_aesthetic1_pickscore0_multireward',
        'code4_greedy_b5_aesthetic1_pickscore10_multireward',
        'code4_greedy_b5_aesthetic1_pickscore15_multireward',
        'code4_greedy_b5_aesthetic1_pickscore20_multireward',
        'code4_greedy_b5_aesthetic1_pickscore25_multireward',
        'code4_greedy_b5_aesthetic1_pickscore2_multireward',
        'code4_greedy_b5_aesthetic1_pickscore3_multireward',
        'code4_greedy_b5_aesthetic1_pickscore5_multireward',
        'code4_greedy_b5_aesthetic1_pickscore30_multireward',
        'code4_greedy_b5_aesthetic1_pickscore50_multireward',
        'code4_greedy_b5_aesthetic1_pickscore70_multireward',
        'code4_greedy_b5_aesthetic1_pickscore100_multireward',
        'code4_greedy_b5_aesthetic1_pickscore150_multireward',
        'code4_greedy_b5_aesthetic1_pickscore200_multireward',
        'code4_greedy_b5_aesthetic1_pickscore250_multireward',
        'code4_greedy_b5_aesthetic1_pickscore300_multireward',
        'code4_greedy_b5_aesthetic1_pickscore350_multireward',
        'code4_greedy_b5_aesthetic1_pickscore400_multireward',
        'code4_greedy_b5_aesthetic1_pickscore450_multireward',
        'code4_greedy_b5_aesthetic1_pickscore500_multireward',
        'code4_greedy_b5_aesthetic1_pickscore750_multireward',
        'code4_greedy_b5_aesthetic1_pickscore1000_multireward',
    ]
    source_dirs = [x for x in base_path.joinpath('outputs').iterdir() if Path.is_dir(x) and x.stem != 'plots' and x.stem in d]

    # source_dirs = [x for x in base_path.joinpath('outputs').iterdir() if (Path.is_dir(x) and x.stem != 'plots')]

    # source_dirs = [x for x in base_path.joinpath('outputs').iterdir()\
    #                 if (Path.is_dir(x) and x.stem != 'plots' and 'grad_i2i' not in x.stem and 'code' not in x.stem)]
    
    # source_dirs2 = [x for x in base_path.joinpath('outputs').iterdir()\
    #                 if (Path.is_dir(x) and x.stem != 'plots' and 'c_code' in x.stem)]

    # source_dirs3 = [x for x in base_path.joinpath('outputs').iterdir()\
    #                 if (Path.is_dir(x) and x.stem != 'plots' and 'c_code5' in x.stem)]

    # source_dirs = list(set(source_dirs + source_dirs2) - set(source_dirs3))

    # breakpoint()
    for source_dir in source_dirs:

        # if ('uncond' in source_dir.stem):
        #     continue

        # if (source_dir.stem in perf.keys()) and ('clipwinrate' in perf[source_dir.stem].keys()):
        #         continue

        scorer = source_dir.stem.split('_')[-1] if 'gs' not in source_dir.stem else source_dir.stem.split('_')[-2]

        if scorer not in _SCORERS:
            continue
        
        clip_winrate = []
        clip_scores = []

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            # prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            prompt_dirs = [x for x in target_dir.iterdir() if Path.is_dir(x)]
            
            for prompt_dir in prompt_dirs:
                
                # Compute clipscores
                # json_path = target_dir.joinpath("images").joinpath(f"{prompt_dir.stem}.json")
                json_path = target_dir.joinpath(f"{prompt_dir.stem}.json")
                captions = {x.stem: prompt_dir.stem for x in prompt_dir.iterdir() if x.suffix == '.png'}
                with open(json_path, 'w') as fp:
                    json.dump(captions, fp)
                
                try:
                    store_path = prompt_dir.joinpath('clipscores.json')
                    score = os.popen(f"python clipscore.py '{json_path.as_posix()}' '{prompt_dir.as_posix()}' --save_per_instance '{store_path.as_posix()}'").read()
                    clip_scores.append(float(score.split(': ')[1]))
                except:
                    print(prompt_dir)
                    print(json_path)
                    raise ValueError()
                
                # Compute clipscore-based win-rate
                with open(prompt_dir.joinpath("clipscores.json"), 'r') as fp:
                    prompt_clipscores = json.load(fp)

                image_ids = [x.stem for x in prompt_dir.iterdir() if x.suffix == '.png']
                prompt_clipscores = np.array([prompt_clipscores[image_id]['CLIPScore'] for image_id in image_ids])
            
                clip_winrate.append((prompt_clipscores > uncond_clipscores[scorer][target_dir.stem][prompt_dir.stem][:len(prompt_clipscores)]).astype(int).sum() / len(prompt_clipscores))

        if source_dir.stem not in perf.keys():
            perf[source_dir.stem] = dict()

        perf[source_dir.stem]['clipscore'] = sum(clip_scores)/len(clip_scores)
        perf[source_dir.stem]['clipwinrate'] = sum(clip_winrate)/len(clip_winrate)

        with open(base_path.joinpath(filename), 'w') as fp:
            json.dump(perf, fp,indent=4)

if __name__ == '__main__':
    compute_clipscore()