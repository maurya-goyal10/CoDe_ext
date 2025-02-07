import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

def compute_t2i(task):

    export_path = Path('outputs/plots').joinpath(task)
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    source_parent = Path('outputs')
    source_dirs = [x for x in source_parent.iterdir() if Path.is_dir(x)]

    perf = dict()

    for source_dir in source_dirs:

        if task not in source_dir.stem:
            continue

        if ('ibon20_' not in source_dir.stem) or ('i2i' in source_dir.stem):
            continue

        print(f'source_dir {source_dir}')

        method_reward = []

        prompt_dirs = [x for x in source_dir.joinpath("images").iterdir() if Path.is_dir(x)]

        rewards = []

        for prompt_dir in prompt_dirs:

            # print(f'prompt_dir {prompt_dir}')

            with open(prompt_dir.joinpath("rewards.json"), 'r') as fp:
                prompt_reward = json.load(fp)

            rewards.extend([np.mean(prompt_reward)])

        method_reward.extend(rewards)

        key = source_dir.stem.split('_')[1]

        # print(key)

        perf[key] = method_reward # np.mean(method_reward) # .split('_')[3]

    # if method == 'i2i_r':
    #     method = 'i2i'
        
    with open(export_path.joinpath(f'ibon20_{task}.json'), 'w') as fp:
        json.dump(dict(sorted(perf.items())), fp)

def compute_fid_t2i(task):

    uncond_path = Path(f'/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/outputs/ibon/uncond_{task}')

    export_path = Path('outputs/plots').joinpath(task)
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    source_parent = Path('outputs')
    source_dirs = [x for x in source_parent.iterdir() if Path.is_dir(x)]

    perf = dict()

    for source_dir in source_dirs:

        if task not in source_dir.stem:
            continue

        if ('ibon20_' not in source_dir.stem) or ('i2i' in source_dir.stem):
            continue

        print(f'source_dir {source_dir}')

        method_reward = []

        prompt_dirs = [x for x in source_dir.joinpath("images").iterdir() if Path.is_dir(x)]

        rewards = []

        for prompt_dir in prompt_dirs:

            # print(f'prompt_dir {prompt_dir}')

            uncond_path_p = uncond_path.joinpath(f'images/{prompt_dir.stem}')

            out = os.popen(f"python /tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/pytorch-fid/src/pytorch_fid/fid_score.py {uncond_path_p.as_posix()} {prompt_dir.as_posix()}").read()

            print(out)

            rewards.append(float(out.split('  ')[-1].split('\n')[0]))

        method_reward.extend(rewards)

        key = source_dir.stem.split('_')[1]

        # print(key)

        perf[key] = method_reward # np.mean(method_reward) # .split('_')[3]

    # if method == 'i2i_r':
    #     method = 'i2i'
        
    with open(export_path.joinpath(f'ibon20_{task}_fid.json'), 'w') as fp:
        json.dump(dict(sorted(perf.items())), fp)

def compute_cmmd_t2i(task):

    uncond_path = Path(f'/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/BoN/outputs/uncond_{task}')

    export_path = Path('outputs/plots').joinpath(task)
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    source_parent = Path('outputs')
    source_dirs = [x for x in source_parent.iterdir() if Path.is_dir(x)]

    perf = dict()

    for source_dir in source_dirs:

        if task not in source_dir.stem:
            continue

        if ('bon' not in source_dir.stem) or ('ibon' in source_dir.stem) or ('i2i' in source_dir.stem):
            continue

        print(f'source_dir {source_dir}')

        method_reward = []

        prompt_dirs = [x for x in source_dir.joinpath("images").iterdir() if Path.is_dir(x)]

        rewards = []

        for prompt_dir in prompt_dirs:

            print(f'prompt_dir {prompt_dir.stem}')

            uncond_path_p = uncond_path.joinpath(f'images/{prompt_dir.stem}')

            out = os.popen(f"python /tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/cmmd-pytorch/main.py {uncond_path_p.as_posix()} {prompt_dir.as_posix()}").read()

            print(out)

            rewards.append(float(out.split('  ')[-1].split('\n')[0]))

        method_reward.extend(rewards)

        key = source_dir.stem.split('_')[0]

        # print(key)

        perf[key] = method_reward # np.mean(method_reward) # .split('_')[3]

    # if method == 'i2i_r':
    #     method = 'i2i'
        
    with open(export_path.joinpath(f'bon_{task}_cmmd.json'), 'w') as fp:
        json.dump(dict(sorted(perf.items())), fp)

def compute(task):

    export_path = Path('outputs/plots').joinpath(task)
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    source_parent = Path('outputs')
    source_dirs = [x for x in source_parent.iterdir() if Path.is_dir(x)]

    perf = dict()

    for source_dir in source_dirs:

        if task not in source_dir.stem:
            continue

        if f'i2i_r' not in source_dir.stem:
            continue

        # print(f'source_dir {source_dir}')

        method_reward = []

        target_dirs = [x for x in source_dir.iterdir() if Path.is_dir(x)]

        for target_dir in target_dirs:

            # print(f'target_dir {target_dir}')

            prompt_dirs = [x for x in target_dir.joinpath("images").iterdir() if Path.is_dir(x)]

            rewards = []

            for prompt_dir in prompt_dirs:

                # print(f'prompt_dir {prompt_dir}')

                with open(prompt_dir.joinpath("rewards.json"), 'r') as fp:
                    prompt_reward = json.load(fp)

                rewards.extend([np.mean(prompt_reward)])

            method_reward.extend(rewards)

        key = source_dir.stem.split('_')[1]

        # print(key)

        perf[key] = method_reward # np.mean(method_reward) # .split('_')[3]

    # if method == 'i2i_r':
    #     method = 'i2i'
        
    with open(export_path.joinpath(f'i2i_metric_pw.json'), 'w') as fp:
        json.dump(dict(sorted(perf.items())), fp)

def plot_point(task):

    export_path = Path('outputs/plots').joinpath(task)
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    methods = {
        'Uncond': 'uncond_metric',
        'CoDe (n=5)': 'ibon5_metric',
        'CoDe (n=100)': 'ibon100_metric',
        'SDEdit': 'i2i_metric',
        'SDEdit + CoDe (n=5)': 'ibon_i2i5_metric',
        'SDEdit + CoDe (n=100)': 'ibon_i2i100_metric',
    }

    result = {
        'method': [],
        'rewards': []
    }

    for method in methods.keys():

        with open(export_path.joinpath(f'{methods[method]}_ext.json'), 'r') as fp:
            perf = json.load(fp)

        if method in ['SDEdit', 'SDEdit + CoDe (n=5)', 'SDEdit + CoDe (n=100)']:
            # max_idx = np.argmax(np.array(perf.values()))
            # result[method] = 1.0/-list(perf.values())[max_idx]
            # print(list(perf.keys())[max_idx])

            rewards = perf['r4'] # [1/(-1*x) for x in perf['r4']]
            if task == 'facedetector':
                rewards = [1/(-1*x) for x in rewards]
            result['rewards'].extend(rewards)
            result['method'].extend([method]* len(rewards))

        elif method in ['CoDe (n=5)', 'CoDe (n=100)', 'Uncond']:
            rewards = perf[methods[method].split('_')[0]] # [1/(-1*x) for x in perf[methods[method].split('_')[0] + '_b5_{task}']]
            if task == 'facedetector':
                rewards = [1/(-1*x) for x in rewards]
            result['rewards'].extend(rewards)
            result['method'].extend([method]* len(rewards))

    # result = dict(sorted(result.items()))
    # print(len(result['method']))
    # print(len(result['rewards']))

    result = pd.DataFrame(result)
    # result = result.T
    # result = result.reset_index()
    # result = result.rename(columns = {'index':'Guidance'})

    # print(result.head())
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.pointplot(result, x='method', y='rewards', errorbar='sd', ax=ax, linestyle='', marker='D', capsize=.1, markersize=10)
    # b = sns.barplot(result, x='method', y='rewards', ax=ax, errorbar='sd')
    # b.bar_label(b.containers[0])

    plt.xlabel('Guidance', fontsize=10)
    plt.ylabel('Rewards', fontsize=10)
    # b.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(export_path.joinpath(f'pointplot_{task}2.png'), dpi=300)

def plot_noise_ratio(task):

    export_path = Path('outputs/plots').joinpath(task)
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    methods = {
        'GeNIe': 'i2i_metric',
        'GeNIe + CD (n=5)': 'ibon_i2i5_metric',
        'GeNIe + CD (n=100)': 'ibon_i2i100_metric',
    }

    result = {
        'method': [],
        'rewards': [],
        'rvals': []
    }

    for method in methods.keys():

        with open(export_path.joinpath(f'{methods[method]}_ext.json'), 'r') as fp:
            perf = json.load(fp)

        for rval in perf.keys():
            rewards = perf[rval]
            if task == 'facedetector':
                rewards = [1/(-1*x) for x in rewards]
            result['rewards'].extend(rewards)
            result['rvals'].extend([round(int(rval[-1])*0.1, 1)]* len(rewards))
            result['method'].extend([method]* len(rewards))

    result = pd.DataFrame(result)
    # result = result.T
    # result = result.reset_index()
    # result = result.rename(columns = {'index':'Guidance'})

    # print(result.head())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(result, x='rvals', y='rewards', hue='method', errorbar='sd', ax=ax, 
                  linestyle='', marker='D', capsize=.1, markersize=10, dodge=.4)
    # b = sns.barplot(result, x='method', y='rewards', ax=ax, errorbar='sd')
    # b.bar_label(b.containers[0])

    plt.xlabel('Noise ratio', fontsize=10)
    plt.ylabel('Rewards', fontsize=10)
    plt.legend(title='Method')
    # b.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(export_path.joinpath(f'noise_{task}.png'), dpi=300)

def plot_rew_txt(task):

    clip_path = Path('/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/clipscore')

    export_path = Path('outputs/plots').joinpath(task)
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    methods = {
        'GeNIe': 'i2i_metric',
        # 'GeNIe + CD (n=5)': 'ibon_i2i5_metric',
        # 'GeNIe + CD (n=100)': 'ibon_i2i100_metric',
    }

    result = {
        # 'method': [],
        'rewards': [],
        'rvals': [],
        'txt_align': []
    }

    for method in methods.keys():

        with open(export_path.joinpath(f'{methods[method]}_ext.json'), 'r') as fp:
            perf = json.load(fp)

        for rval in perf.keys():
            rewards = perf[rval]
            if task == 'facedetector':
                rewards = [1/(-1*x) for x in rewards]
            rewards = [np.mean(rewards)]

            result['rewards'].extend(rewards)
            result['rvals'].extend([round(int(rval[-1])*0.1, 1)]* len(rewards))
            # result['method'].extend([method]* len(rewards))

        with open(clip_path.joinpath(f'{methods[method]}_{task}.json'), 'r') as fp:
            perf = json.load(fp)

        for rval in perf.keys():
            rewards = perf[rval]
            rewards = [np.mean(rewards)]

            result['txt_align'].extend(rewards)

        result = pd.DataFrame(result)

        test_data_melted = pd.melt(result, id_vars='rvals',\
                           var_name="score_type", value_name="value_numbers")
        
        mask = test_data_melted.score_type.isin(['txt_align'])
        scale = test_data_melted[~mask].value_numbers.mean() / test_data_melted[mask].value_numbers.mean()
        test_data_melted.loc[mask, 'value_numbers'] = test_data_melted.loc[mask, 'value_numbers']*scale
        # print(test_data_melted)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        # g = sns.pointplot(x='rvals', y="value_numbers", hue="score_type",\
        #                 data=test_data_melted, ax=ax, legend=False, 
        #                 errorbar='sd', linestyle='-', marker='D', capsize=.1, markersize=10, dodge=.2)
        
        g = sns.pointplot(x='rvals', y="value_numbers", hue="score_type",\
                        data=test_data_melted, ax=ax, legend=False, 
                        linestyle='-', marker='D', capsize=.1, markersize=10)
        
        ax.set_xlabel('Strength (Noise Ratio)')

        # Create a second y-axis with the scaled ticks
        ax.set_ylabel('Expected Rewards')
        ax2 = ax.twinx()

        # Ensure ticks occur at the same positions, then modify labels
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticklabels(np.round(ax.get_yticks() / scale , 2))
        ax2.set_ylabel('Text Alignment (Clip Score)')

        # plt.show()

        # fig, ax = plt.subplots(figsize=(10, 5))
        # sns.lineplot(result, x='rewards', y='txt_align', 
        #              hue='method', style='method', ax=ax, linestyle='-', sort=False)
        # sns.lineplot(result, x='rewards', y='rewards', 
        #              hue='method', style='method', ax=ax, linestyle='-', sort=False)
        # # b = sns.barplot(result, x='method', y='rewards', ax=ax, errorbar='sd')
        # # b.bar_label(b.containers[0])

        # plt.xlabel('Expected Rewards', fontsize=10)
        # plt.ylabel('Text Alignment (ClipScore)', fontsize=10)
        # plt.legend(title='Method')
        # # b.tick_params(labelsize=10)

        plt.tight_layout()
        plt.savefig(export_path.joinpath(f'{method}_{task}.png'), dpi=300)
        plt.close()


if __name__ == '__main__':

    task = 'styletransfer'

    # compute(task)
    plot_point(task)
    # plot_noise_ratio(task)
    # plot_rew_txt(task)
    # compute_t2i(task)
    # compute_cmmd_t2i(task)