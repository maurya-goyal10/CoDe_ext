import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.auto import tqdm

_SCORERS = ['facedetector', 'styletransfer']

# def create_json():

#     source_dirs = [x for x in Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN/outputs').iterdir() if (Path.is_dir(x) and x.stem != 'plots')]
#     for source_dir in source_dirs:

#         scorer = source_dir.stem.split('_')[-1]

#         if scorer not in _SCORERS:
#             continue

#         target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
#         for target_dir in target_dirs:

#             prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
#             for prompt_dir in prompt_dirs:
#                 captions = {x.stem: prompt_dir.stem for x in prompt_dir.iterdir() if x.suffix == '.png'}
#                 with open(target_dir.joinpath("images").joinpath(f'{prompt_dir.stem}.json'), 'w') as fp:
#                     json.dump(captions, fp)

def compute_clipscore():

    # base_path = Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN')
    base_path = Path('/home/sayak/Projects/PhD_GuidedDiff/BoN')

    # Load unconditional rewards
    uncond_clipscores = dict()
    for scorer in _SCORERS:

        scorer_path = base_path.joinpath('outputs').joinpath(f'uncond_{scorer}')

        uncond_clipscores[scorer] = dict()

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

    perf = dict()

    # if Path.exists(base_path.joinpath('perf.json')):
    #     with open(base_path.joinpath('perf.json'), 'r') as fp:
    #         perf = json.load(fp)

    source_dirs = [x for x in base_path.joinpath('outputs').iterdir() if (Path.is_dir(x) and x.stem != 'plots')]
    for source_dir in source_dirs:

        if not 'bon30_' in source_dir.stem:
            continue

        if (source_dir.stem in perf.keys()) and ('clipwinrate' in perf[source_dir.stem].keys()):
                continue

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

        with open(base_path.joinpath('perf_check.json'), 'w') as fp:
            json.dump(perf, fp)

if __name__ == '__main__':
    compute_clipscore()