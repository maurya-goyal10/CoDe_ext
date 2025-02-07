import re
import json

from tqdm.auto import tqdm
from pathlib import Path

_SCORERS = {
    'facedetector': [
        "Headshot of a person with blonde hair with space background",
        "A headshot of a woman looking like a lara croft",
        "Headshot of a woman made of marble",
    ], 
    'styletransfer': [
        "A colorful photo of a eiffel tower",
        "A fantasy photo of a lonely road",
        "portrait of a woman",
        "A fantasy photo of volcanoes",
    ]
}

target_imgs = {
    'facedetector': [
        'og_img_4',
        'og_img_6',
        'og_img_8',
    ],
    'styletransfer': [
        'style_0',
        'style_1',
        'style_2',
    ]
}

def main():

    pending_runs = dict()

    counter = dict()

    source_dirs = [x for x in Path('/tudelft.net/staff-bulk/ewi/insy/VisionLab/smukherjee/PhD_GuidedDiff/BoN/outputs').iterdir() if (Path.is_dir(x) and x.stem != 'plots')]

    for source_dir in tqdm(source_dirs):

        # if ('bon' not in source_dir.stem) or ('ibon' in source_dir.stem) or ('bon_i2i' in source_dir.stem):
        #     continue

        if ('code' not in source_dir.stem) or ('b1_' in source_dir.stem):
            continue

        scorer = source_dir.stem.split('_')[-1]

        if scorer not in _SCORERS.keys():
            continue

        print(source_dir.stem)

        if scorer not in counter.keys():
            counter[scorer] = {
                'total': 0,
                'complete': 0
            }

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]
        for target_dir in target_dirs:

            prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
            for prompt_dir in prompt_dirs:

                num_images = len([x for x in prompt_dir.iterdir() if (Path.is_file(x) and x.suffix == '.png')])
                # if num_images < 50:

                if source_dir.stem not in pending_runs.keys():
                    pending_runs[source_dir.stem] = dict()

                if target_dir.stem not in pending_runs[source_dir.stem].keys():
                    pending_runs[source_dir.stem][target_dir.stem] = dict()

                pending_runs[source_dir.stem][target_dir.stem][prompt_dir.stem] = 50 - num_images

                counter[scorer]['complete'] += num_images
                counter[scorer]['total'] += 50

            # pending prompts
            for prompt in _SCORERS[scorer]:

                if source_dir.stem not in pending_runs.keys():
                    pending_runs[source_dir.stem] = dict()

                if target_dir.stem not in pending_runs[source_dir.stem].keys():
                    pending_runs[source_dir.stem][target_dir.stem] = dict()

                if prompt not in pending_runs[source_dir.stem][target_dir.stem].keys():
                    pending_runs[source_dir.stem][target_dir.stem][prompt] = 50

        # pending targets
        for target in target_imgs[scorer]:

            if source_dir.stem not in pending_runs.keys():
                pending_runs[source_dir.stem] = dict()

            if target not in pending_runs[source_dir.stem].keys():
                pending_runs[source_dir.stem][target] = dict()

            # pending prompts
            for prompt in _SCORERS[scorer]:
                if prompt not in pending_runs[source_dir.stem][target].keys():
                    pending_runs[source_dir.stem][target][prompt] = 50
                    counter[scorer]['total'] += num_images

    with open('status.json', 'w') as fp:
        json.dump(pending_runs, fp=fp, indent=4)

    print(counter)
    for keys in counter.keys():
        print(f'{keys}: {round((counter[keys]["complete"]/counter[keys]["total"] * 100),2)}')

if __name__ == '__main__':
    main()