import os
import json
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

_SCORERS = ["strokegen", "facedetector"] # ['compress'] # ["strokegen", "facedetector", 'styletransfer']
_SCORERS_db = ["/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/assets/face_data/celeb", '/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/assets/style_folder/styles'] # ['/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/assets/compressibility/compress'] # ['/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/assets/stroke_gen/strokes', "/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/assets/face_data/celeb", '/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/assets/style_folder/styles'] 

def main():

    currhost = os.uname()[1]
    root_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff') if "housky" in currhost\
                    else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff')
    outputs_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/BoN/outputs') if "housky" in currhost\
                    else Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN/outputs')

    embeds_path = Path('/glb/data/ptxd_dash/nlasqh/PhD_GuidedDiff/CSD/embeddings')

    # method_bests = ['uncond_strokegen']
            	    # ['c_code40_b5_r6_strokegen', 'grad12999999999999998_styletransfer', 'c_code100_b5_r6_styletransfer', 'grad13_strokegen', 'code40_b5_strokegen', \
                    # 'bon40_strokegen', 'code40_b1_styletransfer', 'code40_b1_strokegen', 'bon40_styletransfer', 'code100_b5_styletransfer']
                    
    d = ['c_code_10_b1_r6_styletransfer',
        'c_code_10_b1_r7_facedetector',
        'c_code_20_b1_r6_styletransfer',
        'c_code_20_b1_r7_facedetector',
        'c_code_30_b1_r6_styletransfer',
        'c_code_30_b1_r7_facedetector']
    

    # source_dirs = [x for x in base_path.joinpath('outputs').iterdir() if Path.is_dir(x) and x.stem != 'plots' and x.stem in d]

    
    source_dirs = [x for x in outputs_path.iterdir()\
                    if (Path.is_dir(x) and x.stem != 'plots' and x.stem in d)]

    # breakpoint()

    # To extract the reference image embeddings
    # for scorer, scorer_db in zip(_SCORERS, _SCORERS_db):
    #     out = os.popen(f"python main_sim.py --dataset custom -a vit_large --pt_style vgg --feattype gram \
    #                 --gram_dims 1024 --world-size 1 --dist-url tcp://localhost:6001 -b 128 -j 8 --embed_dir ./embeddings/embeddings_{scorer} \
    #                 --data-dir '{scorer_db}' --model_path pretrainedmodels/pytorch_model.bin").read()

    
    # source_dirs = [x for x in base_path.joinpath('outputs').iterdir()\
    #                 if (Path.is_dir(x) and x.stem != 'plots' and 'i2i' not in x.stem)]
    
    # source_dirs2 = [x for x in base_path.joinpath('outputs').iterdir()\
    #                 if (Path.is_dir(x) and x.stem != 'plots' and 'c_code' in x.stem)]

    # source_dirs3 = [x for x in base_path.joinpath('outputs').iterdir()\
    #                 if (Path.is_dir(x) and x.stem != 'plots' and 'c_code5' in x.stem)]

    # source_dirs = list(set(source_dirs + source_dirs2) - set(source_dirs3))                    

    # Compute scores
    perf = dict()

    if Path.exists(Path('perf_gram_ccode_b1.json')):
        with open('perf_gram_ccode_b1.json', 'r') as fp:
            perf = json.load(fp)


    for i, source_dir in enumerate(source_dirs):
        prog = 100 * (i/len(source_dirs))
        print(f"I am at {prog}%")


        if source_dir.stem in perf.keys():
            continue

        scorer = source_dir.stem.split('_')[-1]

        if scorer not in _SCORERS:
            continue

        exp_rew = []
        win_rate = []

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            for prompt_dir in prompt_dirs:

                embed_dir = embeds_path.joinpath(f"embeddings_{source_dir.stem}/{target_dir.stem}/{prompt_dir.stem}")
                try: 
                    
                    os.makedirs(embed_dir)
            
                    out = os.popen(f"python main_sim.py --dataset custom -a vit_large --pt_style vgg --feattype gram \
                    --gram_dims 1024 --world-size 1 --dist-url tcp://localhost:6001 -b 128 -j 8 \
                    --embed_dir '{embed_dir}' \
                    --data-dir '{prompt_dir}' --model_path pretrainedmodels/pytorch_model.bin").read()
                    
                    # breakpoint()
                    db = np.load(f'./embeddings/embeddings_{scorer}/vgg_vit_large_custom_gram_1024_query_/1/database/embeddings_0.pkl', allow_pickle=True)
                    # if 'style' in scorer:
                    #     db = np.load(f'./embeddings/embeddings_{scorer}/vgg_vit_large_custom_gram_1024_query_/1/database/embeddings_0.pkl', allow_pickle=True)
                    # else:
                    #     db = np.load(f'./embeddings/embeddings_{scorer}/vgg_vit_large_custom_gram_1024_query_/1/database/embeddings_0.pkl', allow_pickle=True)
                    
                    query = np.load(f'{embed_dir}/vgg_vit_large_custom_gram_1024_query_/1/database/embeddings_0.pkl', allow_pickle=True)

                    print(target_dir)
                    score = query['embeddings'] @ db['embeddings'][db['filenames'].index(target_dir.stem)].T
                    exp_rew.extend(score)

                    # Saving per prompt set of scores
                    with open(f"{embed_dir}/reward_gram.json", "w") as final:
                        json.dump(score.tolist(), final)
                
                except: 
                    
                    print("Folder already exists")
                    with open(f"{embed_dir}/reward_gram.json", "r") as final:
                        score = json.load(final)
                        exp_rew.extend(score)
                
        if source_dir.stem not in perf.keys():
            perf[source_dir.stem] = dict()

        perf[source_dir.stem]['exp_rew'] = sum(exp_rew)/len(exp_rew)

        with open('perf_gram_ccode_b1.json', 'w') as fp:
            json.dump(perf, fp)



if __name__ == '__main__':
    main()