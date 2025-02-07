import os
import json
import numpy as np
# import seaborn as sns
import pandas as pd
# import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm

_SCORERS = ['strokegen', 'facedetector'] # ['compress'] # ['strokegen', 'facedetector', 'styletransfer'] 

def compute_clipscore():

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff')
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext')
    base_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN') if "housky" in currhost\
                    # else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN')
                    else Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/BoN')
                    

    perf = dict()

    if Path.exists(base_path.joinpath('perf_ccode_b1_clipscore.json')):
        with open(base_path.joinpath('perf_ccode_b1_clipscore.json'), 'r') as fp:
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

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]

            for prompt_dir in prompt_dirs:

                if not Path.exists(prompt_dir.joinpath('clipscores.json')):

                    json_path = target_dir.joinpath("images").joinpath(f"{prompt_dir.stem}.json")
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
    
    d = ['c_code_10_b1_r6_styletransfer',
        'c_code_10_b1_r7_facedetector',
        'c_code_20_b1_r6_styletransfer',
        'c_code_20_b1_r7_facedetector',
        'c_code_30_b1_r6_styletransfer',
        'c_code_30_b1_r7_facedetector']
    

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

        scorer = source_dir.stem.split('_')[-1]

        if scorer not in _SCORERS:
            continue
        
        clip_winrate = []
        clip_scores = []

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            for prompt_dir in prompt_dirs:
                
                # Compute clipscores
                json_path = target_dir.joinpath("images").joinpath(f"{prompt_dir.stem}.json")
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

        with open(base_path.joinpath('perf_ccode_b1_clipscore.json'), 'w') as fp:
            json.dump(perf, fp)

if __name__ == '__main__':
    compute_clipscore()