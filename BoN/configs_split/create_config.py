import os
import copy
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf

_METHODS = ['code_grad_final_general'] # ['c_bon', 'i2i'] # 'ibon', ibon_i2i', 'bon', 'uncond', 'i2i', 'bon_i2i', 'code', 'c_code', 'grad_i2i_mpgd', 'grad', 'code_grad', 'code', 'code_grad'

_SCORERS = {
    'aesthetic': '../assets/eval_simple_animals.txt', 
    # 'hpsv2': '../assets/hps_v2_all_eval.txt', 
    # 'facedetector': '../assets/face.txt', 
    # 'styletransfer': '../assets/style.txt',
    # 'strokegen': '../assets/stroke.txt',
    # 'compress': '../assets/compressibility.txt',
    # 'imagereward': '../assets/hps_v2_all_eval.txt', 
    # 'pickscore': '../assets/hps_v2_all_eval.txt',
    # 'multireward': None,
}

def create_function():

    currhost = os.uname()[1]
    template = OmegaConf.load('template_shell.yaml') if "housky" in currhost else OmegaConf.load('template.yaml') 

    config_dir = Path('.')

    for method in _METHODS:

        curr_path = config_dir.joinpath(method)

        if not Path.exists(curr_path):
            Path.mkdir(curr_path, parents=True)

        for scorer in _SCORERS.keys():
            
            # if scorer ==  "aesthetic":
            #     continue

            num_prompts = 6 #if scorer == 'facedetector' else 4
            num_targets = 1 # if scorer == 'strokegen' else 3

            print(f'{method} {scorer}')

            if method == 'uncond' or method=='uncond2':

                for prompt_idx in range(num_prompts):

                    for target_idx in range(num_targets):

                        curr_config = copy.deepcopy(template)
                        curr_config.project.name = f'{method}_{scorer}'
                        curr_config.project.promptspath = _SCORERS[scorer]

                        curr_config.guidance.method = method
                        curr_config.guidance.scorer = scorer
                        curr_config.guidance.target_idxs = [target_idx]
                        curr_config.guidance.prompt_idxs = [prompt_idx]
                        if "target_idxs" in curr_config.guidance:
                            del curr_config.guidance["target_idxs"]
                        curr_config.guidance.block_size = 5
                        curr_config.guidance.num_images_per_prompt = 10
                        curr_config.guidance.num_gen_target_images_per_prompt = 10
                        

                        filename = f'{method}_p{prompt_idx}_{scorer}'
                        # filename = f'{method}_p{prompt_idx}_t{target_idx}_{scorer}'
                        savepath = curr_path.joinpath(f'{filename}.yaml')
                        OmegaConf.save(curr_config, savepath)
            
            elif method in ['code_b1']:

                for num_samples in [40]: # [10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for block_size in [1]: # [5, 10, 20, 50, 100]

                        for prompt_idx in range(num_prompts):

                            if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    m = method.split('_')[0]
                                    curr_config.project.name = f'{m}{num_samples}_b{block_size}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    
                                    if "target_idxs" in curr_config.guidance:
                                        del curr_config.guidance["target_idxs"]

                                    filename = f'{m}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)
                            
                            else:
                                curr_config = copy.deepcopy(template)
                                m = method.split('_')[0]
                                curr_config.project.name = f'{m}{num_samples}_t{curr_config.guidance.num_inference_steps}_b{block_size}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.block_size = block_size
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                curr_config.guidance.num_inference_steps = 100
                                curr_config.guidance.num_images_per_prompt = 10
                                curr_config.guidance.sampling = "greedy"
                                
                                if "target_idxs" in curr_config.guidance:
                                    del curr_config.guidance["target_idxs"]

                                filename = f'{m}{num_samples}_t{curr_config.guidance.num_inference_steps}_p{prompt_idx}_b{block_size}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)


            elif method in ['c_code_b1']:

                for num_samples in [10, 20, 30]: # [10, 20, 30, 40]: # [25, 50, 100, 200, 500]:
                    
                    pc = [0.7] if 'face' in scorer else [0.6]
                    for percent_noise in pc:
                        for block_size in [1]: # [5, 10, 20, 50, 100]

                            for prompt_idx in range(num_prompts):

                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    m = 'c_code'
                                    curr_config.project.name = f'{m}_{num_samples}_b{block_size}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    curr_config.guidance.percent_noise = float(round(percent_noise,1))

                                    filename = f'{m}_{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

                                
            elif method in ['code']:

                for num_samples in [4]:# ['var40','var40i']: # ['var4i','var4','var4rev']: # [4,'var4']: #[10, 20, 30, 40]: # [10, 20, 30, 40]: [4,40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]

                        for prompt_idx in range(num_prompts):
                            
                            for sampling in ["greedy"]:# ["greedy","multinomial"]:
                                
                                for temp in [1000]:#[2000,3000,4000,6000]: # [200,200000]
                                
                                    if sampling == "greedy":
                                        temp = None

                                    if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                        for target_idx in range(num_targets):

                                            curr_config = copy.deepcopy(template)
                                            curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_{scorer}'
                                            curr_config.project.promptspath = _SCORERS[scorer]

                                            curr_config.guidance.method = method
                                            curr_config.guidance.scorer = scorer
                                            curr_config.guidance.num_samples = num_samples
                                            curr_config.guidance.block_size = block_size
                                            curr_config.guidance.target_idxs = [target_idx]
                                            curr_config.guidance.prompt_idxs = [prompt_idx]
                                            curr_config.guidance.sampling = sampling
                                            if num_samples == "var4":
                                                curr_config.guidance.samples_schedule = [10, 6, 6, 4, 4, 2, 2, 2, 2, 2]
                                            elif num_samples == "var4i":
                                                curr_config.guidance.samples_schedule = [6, 6, 4, 4, 4, 4, 4, 2, 2, 2]
                                            elif num_samples == "var4rev":
                                                curr_config.guidance.samples_schedule = [10, 6, 6, 4, 4, 2, 2, 2, 2, 2][::-1]
                                            
                                            filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                            
                                            if sampling != "greedy":
                                                curr_config.guidance.temp = temp
                                                curr_config.project.name = f'{method}{num_samples}_{sampling}_temp{temp}_b{block_size}_{scorer}'
                                                filename = f'{method}{num_samples}_{sampling}_temp{temp}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                                
                                            
                                            savepath = curr_path.joinpath(f'{filename}.yaml')
                                            OmegaConf.save(curr_config, savepath)
                                            
                                    elif scorer == "multireward":
                                        scorer1 = "aesthetic"
                                        scorer2 = "pickscore"
                                        
                                        curr_config = copy.deepcopy(template)
                                        curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_{scorer}_{scorer2}_{scorer1}'
                                        
                                        curr_config.guidance.scorer1 =  scorer1
                                        curr_config.guidance.scorer2 =  scorer2
                                        curr_config.guidance.scorer_weight = 1
                                        
                                        curr_config.project.promptspath = _SCORERS[scorer1]

                                        curr_config.guidance.method = method
                                        curr_config.guidance.scorer = scorer
                                        curr_config.guidance.scorer = scorer
                                        curr_config.guidance.scorer = scorer
                                        curr_config.guidance.num_samples = num_samples
                                        curr_config.guidance.block_size = block_size
                                        curr_config.guidance.prompt_idxs = [prompt_idx]
                                        curr_config.guidance.num_images_per_prompt = 10
                                        curr_config.guidance.num_gen_target_images_per_prompt = 10
                                        curr_config.guidance.sampling = sampling
                                        if num_samples == "var4":
                                            curr_config.guidance.samples_schedule = [10, 6, 6, 4, 4, 2, 2, 2, 2, 2]
                                        elif num_samples == "var4i":
                                                curr_config.guidance.samples_schedule = [6, 6, 4, 4, 4, 4, 4, 2, 2, 2]
                                        elif num_samples == "var40":
                                                curr_config.guidance.samples_schedule = [64, 64, 64, 64, 64, 32, 16, 16, 8, 8]
                                        elif num_samples == "var40i":
                                                curr_config.guidance.samples_schedule = [64, 64, 64, 32, 32, 32, 32, 32, 24, 24]
                                        elif num_samples == "var4rev":
                                            curr_config.guidance.samples_schedule = [10, 6, 6, 4, 4, 2, 2, 2, 2, 2][::-1]
                                        
                                        if "target_idxs" in curr_config.guidance:
                                            del curr_config.guidance["target_idxs"]

                                        filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_{scorer}_{scorer2}_{scorer1}'
                                        if sampling != "greedy":
                                            curr_config.guidance.temp = temp
                                            curr_config.project.name = f'{method}{num_samples}_{sampling}_temp{temp}_b{block_size}_{scorer}_{scorer2}_{scorer1}'   
                                            filename = f'{method}{num_samples}_{sampling}_temp{temp}_p{prompt_idx}_b{block_size}_{scorer}_{scorer2}_{scorer1}' 
                                            
                                            
                                        savepath = curr_path.joinpath(f'{filename}.yaml')
                                        OmegaConf.save(curr_config, savepath)
                                        

                                    else:
                                        curr_config = copy.deepcopy(template)
                                        curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_{scorer}'
                                        curr_config.project.promptspath = _SCORERS[scorer]

                                        curr_config.guidance.method = method
                                        curr_config.guidance.scorer = scorer
                                        curr_config.guidance.num_samples = num_samples
                                        curr_config.guidance.block_size = block_size
                                        curr_config.guidance.prompt_idxs = [prompt_idx]
                                        curr_config.guidance.num_images_per_prompt = 10
                                        curr_config.guidance.num_gen_target_images_per_prompt = 10
                                        curr_config.guidance.sampling = sampling
                                        if num_samples == "var4":
                                            curr_config.guidance.samples_schedule = [10, 6, 6, 4, 4, 2, 2, 2, 2, 2]
                                        elif num_samples == "var4i":
                                                curr_config.guidance.samples_schedule = [6, 6, 4, 4, 4, 4, 4, 2, 2, 2]
                                        elif num_samples == "var40":
                                                curr_config.guidance.samples_schedule = [64, 64, 64, 64, 64, 32, 16, 16, 8, 8]
                                        elif num_samples == "var40i":
                                                curr_config.guidance.samples_schedule = [64, 64, 64, 32, 32, 32, 32, 32, 24, 24]
                                        elif num_samples == "var4rev":
                                            curr_config.guidance.samples_schedule = [10, 6, 6, 4, 4, 2, 2, 2, 2, 2][::-1]
                                        
                                        if "target_idxs" in curr_config.guidance:
                                            del curr_config.guidance["target_idxs"]

                                        filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_{scorer}'
                                        if sampling != "greedy":
                                            curr_config.guidance.temp = temp
                                            curr_config.project.name = f'{method}{num_samples}_{sampling}_temp{temp}_b{block_size}_{scorer}'   
                                            filename = f'{method}{num_samples}_{sampling}_temp{temp}_p{prompt_idx}_b{block_size}_{scorer}' 
                                            
                                            
                                        savepath = curr_path.joinpath(f'{filename}.yaml')
                                        OmegaConf.save(curr_config, savepath)
                                
            elif method in ['code_ext']:

                for num_samples in [40]: #[10, 20, 30, 40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]

                        for prompt_idx in range(num_prompts):

                            if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    curr_config.project.name = f'{method}{num_samples}_b{block_size}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]

                                    filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

                            else:
                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{num_samples}_b{block_size}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.block_size = block_size
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                curr_config.guidance.num_images_per_prompt = 5
                                curr_config.guidance.num_gen_target_images_per_prompt = 5
                                if "target_idxs" in curr_config.guidance:
                                    del curr_config.guidance["target_idxs"]

                                filename = f'{method}{num_samples}_p{prompt_idx}_b{block_size}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)
                                
            elif method in ['code_grad','code_grad_new','code_grad_new_variant']:

                for num_samples in [4]: #[10, 20, 30, 40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]
                        
                        st = 0.6
                        et = 0.2
                        
                        for guidance_scale in [0.3]: # [0.5,1,10,15]

                            for prompt_idx in range(num_prompts):

                                if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                    for target_idx in range(num_targets):

                                        curr_config = copy.deepcopy(template)
                                        curr_config.project.name = f'{method}{num_samples}_b{block_size}_{scorer}'
                                        curr_config.project.promptspath = _SCORERS[scorer]

                                        curr_config.guidance.method = method
                                        curr_config.guidance.scorer = scorer
                                        curr_config.guidance.num_samples = num_samples
                                        curr_config.guidance.block_size = block_size
                                        curr_config.guidance.target_idxs = [target_idx]
                                        curr_config.guidance.prompt_idxs = [prompt_idx]

                                        filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                        savepath = curr_path.joinpath(f'{filename}.yaml')
                                        OmegaConf.save(curr_config, savepath)

                                else:
                                    curr_config = copy.deepcopy(template)
                                    curr_config.project.name = f'{method}{num_samples}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    curr_config.guidance.guidance_scale = float(guidance_scale)
                                    curr_config.guidance.num_images_per_prompt = 5
                                    curr_config.guidance.num_gen_target_images_per_prompt = 5
                                    curr_config.guidance.start_time = float(st)
                                    curr_config.guidance.end_time = float(et)
                                    if "target_idxs" in curr_config.guidance:
                                        del curr_config.guidance["target_idxs"]

                                    filename = f'{method}{num_samples}_p{prompt_idx}_b{block_size}_{scorer}_st{int(st*10)}_et{int(et*10)}_gs{int(float(guidance_scale)*10)}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)
                                    
            elif method in ['code_grad_final']:
                
                for num_samples in [4]: #[10, 20, 30, 40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]
                        
                        st = 0.7
                        et = 0.3
                        
                        for guidance_scale in [0.3]: # [0.5,1,10,15]

                            for prompt_idx in range(num_prompts):
                                
                                for do_clustering in [False]: # [True, False]
                                    
                                    for clustering_method in ["KMeans"]: #["KMeans", "HDBSCAN"]:
                                        if not do_clustering:
                                            clustering_method = None
                                    
                                        for sampling in ['greedy']: # ['greedy', "multinomial"]
                                            
                                            for temp in [200000,200]:
                                                if(sampling == "greedy"):
                                                    temp = None
                                                    
                                                for guidance_method in ["FreeDoM"]: # ["FreeDoM", "DPS"]:

                                                    if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                                        for target_idx in range(num_targets):

                                                            curr_config = copy.deepcopy(template)
                                                                
                                                            curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_{scorer}'
                                                            curr_config.project.promptspath = _SCORERS[scorer]

                                                            curr_config.guidance.method = method
                                                            curr_config.guidance.scorer = scorer
                                                            curr_config.guidance.num_samples = num_samples
                                                            curr_config.guidance.block_size = block_size
                                                            curr_config.guidance.target_idxs = [target_idx]
                                                            curr_config.guidance.prompt_idxs = [prompt_idx]
                                                            curr_config.guidance.sampling = sampling
                                                            curr_config.guidance.guidance_method = guidance_method

                                                            if temp is not None:
                                                                curr_config.guidance.temp = temp
                                                                curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_temp{temp}_{scorer}'
                                                                filename = f'{method}{num_samples}_{sampling}_temp{temp}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                                            else:
                                                                filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                                            savepath = curr_path.joinpath(f'{filename}.yaml')
                                                            OmegaConf.save(curr_config, savepath)

                                                    else:
                                                        curr_config = copy.deepcopy(template)
                                                        
                                                        curr_config.project.promptspath = _SCORERS[scorer]

                                                        curr_config.guidance.method = method
                                                        curr_config.guidance.scorer = scorer
                                                        curr_config.guidance.num_samples = num_samples
                                                        curr_config.guidance.block_size = block_size
                                                        curr_config.guidance.prompt_idxs = [prompt_idx]
                                                        curr_config.guidance.guidance_scale = float(guidance_scale)
                                                        curr_config.guidance.num_images_per_prompt = 10
                                                        curr_config.guidance.num_gen_target_images_per_prompt = 5
                                                        curr_config.guidance.start_time = float(st)
                                                        curr_config.guidance.end_time = float(et)
                                                        curr_config.guidance.do_clustering = do_clustering
                                                        curr_config.guidance.sampling = sampling
                                                        curr_config.guidance.guidance_method = guidance_method
                                                        
                                                        if "target_idxs" in curr_config.guidance:
                                                            del curr_config.guidance["target_idxs"]

                                                        if do_clustering: 
                                                            curr_config.guidance.clustering_method = clustering_method
                                                            if temp is not None:
                                                                curr_config.guidance.temp = temp
                                                                curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_temp{temp}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_temp{temp}_{scorer}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                            else:
                                                                curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_{scorer}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                                
                                                        else:
                                                            if temp is not None:
                                                                curr_config.guidance.temp = temp
                                                                curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_temp{temp}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_temp{temp}_{scorer}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                            else:
                                                                curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_{scorer}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                            
                                                        savepath = curr_path.joinpath(f'{filename}.yaml')
                                                        OmegaConf.save(curr_config, savepath)

            elif method in ['code_grad_final_general']:
                
                for num_samples in [4]: #[10, 20, 30, 40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]
                        
                        for guidance_blocksize in [5]:
                        
                            st = 0.7
                            et = 0.3
                            
                            for guidance_scale in [0.3]:# [0.3,0.4,0.5,0.7]: # [0.5,1,10,15]

                                for prompt_idx in range(num_prompts):
                                    
                                    for do_clustering in [False]: # [True, False]
                                        
                                        for clustering_method in ["KMeans"]: # ["KMeans", "HDBSCAN"]
                                            if not do_clustering:
                                                clustering_method = None
                                        
                                            for sampling in ['greedy']: # ['greedy', "multinomial"]:
                                                
                                                for temp in [200000,200]:
                                                    if(sampling == "greedy"):
                                                        temp = None
                                                        
                                                    for guidance_method in ["FreeDoM"]:# ["FreeDoM", "DPS"]:

                                                        if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                                            for target_idx in range(num_targets):

                                                                curr_config = copy.deepcopy(template)
                                                                    
                                                                curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_gb{guidance_blocksize}_{scorer}'
                                                                curr_config.project.promptspath = _SCORERS[scorer]

                                                                curr_config.guidance.method = method
                                                                curr_config.guidance.scorer = scorer
                                                                curr_config.guidance.num_samples = num_samples
                                                                curr_config.guidance.block_size = block_size
                                                                curr_config.guidance.target_idxs = [target_idx]
                                                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                                                curr_config.guidance.sampling = sampling
                                                                curr_config.guidance.guidance_method = guidance_method
                                                                curr_config.guidance.guidance_blocksize = guidance_blocksize

                                                                if temp is not None:
                                                                    curr_config.guidance.temp = temp
                                                                    curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_gb{guidance_blocksize}_temp{temp}_{scorer}'
                                                                    filename = f'{method}{num_samples}_{sampling}_temp{temp}_p{prompt_idx}_t{target_idx}_b{block_size}_gb{guidance_blocksize}_{scorer}'
                                                                else:
                                                                    filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_t{target_idx}_b{block_size}_gb{guidance_blocksize}_{scorer}'
                                                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                                                OmegaConf.save(curr_config, savepath)

                                                        else:
                                                            curr_config = copy.deepcopy(template)
                                                            
                                                            curr_config.project.promptspath = _SCORERS[scorer]

                                                            curr_config.guidance.method = method
                                                            curr_config.guidance.scorer = scorer
                                                            curr_config.guidance.num_samples = num_samples
                                                            curr_config.guidance.block_size = block_size
                                                            curr_config.guidance.prompt_idxs = [prompt_idx]
                                                            curr_config.guidance.guidance_scale = float(guidance_scale)
                                                            curr_config.guidance.num_images_per_prompt = 10
                                                            curr_config.guidance.num_gen_target_images_per_prompt = 10
                                                            curr_config.guidance.start_time = float(st)
                                                            curr_config.guidance.end_time = float(et)
                                                            curr_config.guidance.do_clustering = do_clustering
                                                            curr_config.guidance.sampling = sampling
                                                            curr_config.guidance.guidance_method = guidance_method
                                                            curr_config.guidance.guidance_blocksize = guidance_blocksize
                                                            
                                                            if "target_idxs" in curr_config.guidance:
                                                                del curr_config.guidance["target_idxs"]

                                                            if do_clustering: 
                                                                curr_config.guidance.clustering_method = clustering_method
                                                                if temp is not None:
                                                                    curr_config.guidance.temp = temp
                                                                    curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_gb{guidance_blocksize}_temp{temp}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                    filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_gb{guidance_blocksize}_temp{temp}_{scorer}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                                else:
                                                                    curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_gb{guidance_blocksize}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                    filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_gb{guidance_blocksize}_{scorer}_st{int(st*10)}_et{int(et*10)}_{clustering_method}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                                    
                                                            else:
                                                                if temp is not None:
                                                                    curr_config.guidance.temp = temp
                                                                    curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_gb{guidance_blocksize}_temp{temp}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                    filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_gb{guidance_blocksize}_temp{temp}_{scorer}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                                else:
                                                                    curr_config.project.name = f'{method}{num_samples}_{sampling}_b{block_size}_gb{guidance_blocksize}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_{scorer}_gs{int(guidance_scale*10)}'
                                                                    filename = f'{method}{num_samples}_{sampling}_p{prompt_idx}_b{block_size}_gb{guidance_blocksize}_{scorer}_st{int(st*10)}_et{int(et*10)}_{guidance_method}_gs{int(float(guidance_scale)*10)}'
                                                                
                                                            savepath = curr_path.joinpath(f'{filename}.yaml')
                                                            OmegaConf.save(curr_config, savepath)
            elif method in ['code_grad_ext']:

                for num_samples in [4]: #[10, 20, 30, 40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]
                        
                        st = 1.0
                        et = 0.3
                        for grad_block_size in [4]:

                            for guidance_scale in [0.3]: # [0.5,1,10,15]

                                for prompt_idx in range(num_prompts):

                                    if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                        for target_idx in range(num_targets):

                                            curr_config = copy.deepcopy(template)
                                            curr_config.project.name = f'{method}{num_samples}_b{block_size}_{scorer}'
                                            curr_config.project.promptspath = _SCORERS[scorer]

                                            curr_config.guidance.method = method
                                            curr_config.guidance.scorer = scorer
                                            curr_config.guidance.num_samples = num_samples
                                            curr_config.guidance.block_size = block_size
                                            curr_config.guidance.target_idxs = [target_idx]
                                            curr_config.guidance.prompt_idxs = [prompt_idx]

                                            filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                            savepath = curr_path.joinpath(f'{filename}.yaml')
                                            OmegaConf.save(curr_config, savepath)

                                    else:
                                        curr_config = copy.deepcopy(template)
                                        curr_config.project.name = f'{method}{num_samples}_b{block_size}_gb{grad_block_size}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                                        curr_config.project.promptspath = _SCORERS[scorer]

                                        curr_config.guidance.method = method
                                        curr_config.guidance.scorer = scorer
                                        curr_config.guidance.num_samples = num_samples
                                        curr_config.guidance.block_size = block_size
                                        curr_config.guidance.grad_block_size = grad_block_size
                                        curr_config.guidance.prompt_idxs = [prompt_idx]
                                        curr_config.guidance.guidance_scale = float(guidance_scale)
                                        curr_config.guidance.num_images_per_prompt = 5
                                        curr_config.guidance.num_gen_target_images_per_prompt = 5
                                        curr_config.guidance.start_time = float(st)
                                        curr_config.guidance.end_time = float(et)
                                        if "target_idxs" in curr_config.guidance:
                                            del curr_config.guidance["target_idxs"]

                                        filename = f'{method}{num_samples}_p{prompt_idx}_b{block_size}_gb{grad_block_size}_{scorer}_st{int(st*10)}_et{int(et*10)}_gs{int(float(guidance_scale)*10)}'
                                        savepath = curr_path.joinpath(f'{filename}.yaml')
                                        OmegaConf.save(curr_config, savepath)


            elif method == 'bon':

                for num_samples in [10, 20, 30, 100]: # [5, 10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for prompt_idx in range(num_prompts):

                        if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                            for target_idx in range(num_targets):

                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{num_samples}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.target_idxs = [target_idx]
                                curr_config.guidance.prompt_idxs = [prompt_idx]

                                filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)

                        else:
                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}{num_samples}_{scorer}'
                            curr_config.project.promptspath = _SCORERS[scorer]

                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.num_samples = num_samples
                            curr_config.guidance.prompt_idxs = [prompt_idx]

                            filename = f'{method}{num_samples}_p{prompt_idx}_{scorer}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)

            
            elif method == 'c_bon':

                for num_samples in [10, 20, 30, 100]: # [10, 20, 30, 40]: # [5, 10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for percent_noise in [0.8]: #  np.arange(0.5, 0.9, 0.1)
                        for prompt_idx in range(num_prompts):

                            for target_idx in range(num_targets):

                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{num_samples}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.target_idxs = [target_idx]
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                curr_config.guidance.percent_noise = float(round(percent_noise,1))

                                filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_r{float(round(percent_noise,1))}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)
            
            elif method == 'i2i':

                for percent_noise in [0.6, 0.7, 0.8]: # np.arange(0.5, 0.9, 0.1):
                    for prompt_idx in range(num_prompts):

                        for target_idx in range(num_targets):

                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                            curr_config.project.promptspath = _SCORERS[scorer]

                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.target_idxs = [target_idx]
                            curr_config.guidance.prompt_idxs = [prompt_idx]
                            curr_config.guidance.percent_noise = float(round(percent_noise,1))

                            filename = f'{method}_p{prompt_idx}_t{target_idx}_r{float(round(percent_noise,1))}_{scorer}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)

            elif method == 'grad' or method == 'grad_fixed_mpgd':
            
                for guidance_scale in [0.3]: # np.arange(100, 600, 100): # [25, 50, 100, 200, 500]:

                    for prompt_idx in range(num_prompts):
                        st = 0.6
                        et = 0.2

                        if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                            for target_idx in range(num_targets):

                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{int(float(round(guidance_scale,1))*10)}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.guidance_scale = float(guidance_scale)
                                curr_config.guidance.target_idxs = [target_idx]
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                
                                filename = f'{method}{int(float(round(guidance_scale,1))*10)}_p{prompt_idx}_t{target_idx}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)

                        else:
                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}_p{prompt_idx}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)})'
                            curr_config.project.promptspath = _SCORERS[scorer]

                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.guidance_scale = float(guidance_scale)
                            curr_config.guidance.prompt_idxs = [prompt_idx]
                            del(curr_config.guidance["target_idxs"])
                            
                            curr_config.guidance.num_images_per_prompt = 5
                            curr_config.guidance.num_gen_target_images_per_prompt = 5
                            
                            curr_config.guidance.start_time = float(st)
                            curr_config.guidance.end_time = float(et)
                            
                            filename = f'{method}_p{prompt_idx}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)

            elif method == 'grad_fixed':
                for block_size in [5]: # [5, 10, 20, 50, 100]
                    for guidance_scale in [0.3]: # np.arange(100, 600, 100): # [25, 50, 100, 200, 500]:

                        for prompt_idx in range(num_prompts):
                            st = 0.6
                            et = 0.2

                            if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    curr_config.project.name = f'{method}{int(float(round(guidance_scale,1))*10)}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.guidance_scale = float(guidance_scale)
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    
                                    filename = f'{method}{int(float(round(guidance_scale,1))*10)}_p{prompt_idx}_t{target_idx}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

                            else:
                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                                curr_config.project.promptspath = _SCORERS[scorer]
                                
                                curr_config.guidance.block_size = block_size
                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.guidance_scale = float(guidance_scale)
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                del(curr_config.guidance["target_idxs"])
                                
                                curr_config.guidance.num_images_per_prompt = 5
                                curr_config.guidance.num_gen_target_images_per_prompt = 5
                                
                                curr_config.guidance.start_time = float(st)
                                curr_config.guidance.end_time = float(et)
                                
                                filename = f'{method}_p{prompt_idx}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)
                            
            elif method == 'grad_fixed_das':
                for block_size in [1]: # [5, 10, 20, 50, 100]
                    for guidance_scale in [0.5]: # np.arange(100, 600, 100): # [25, 50, 100, 200, 500]:

                        for prompt_idx in range(num_prompts):
                            st = 0.6
                            et = 0.2

                            if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                
                                    # fix the num of inference steps
                                    curr_config.guidance.num_inference_steps = 100
                                    curr_config.scheduler = "ddim"
                                    
                                    
                                    curr_config.project.name = f'{method}{int(float(round(guidance_scale,1))*10)}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.guidance_scale = float(guidance_scale)
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    
                                    filename = f'{method}{int(float(round(guidance_scale,1))*10)}_p{prompt_idx}_t{target_idx}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

                            else:
                                curr_config = copy.deepcopy(template)
                                
                                # fix the num of inference steps
                                curr_config.guidance.num_inference_steps = 100
                                curr_config.scheduler = "ddim"
                                
                                curr_config.project.name = f'{method}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                                curr_config.project.promptspath = _SCORERS[scorer]
                                
                                curr_config.guidance.block_size = block_size
                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.guidance_scale = float(guidance_scale)
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                del(curr_config.guidance["target_idxs"])
                                
                                curr_config.guidance.num_images_per_prompt = 5
                                curr_config.guidance.num_gen_target_images_per_prompt = 5
                                
                                curr_config.guidance.start_time = float(st)
                                curr_config.guidance.end_time = float(et)
                                
                                filename = f'{method}_p{prompt_idx}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                                if curr_config.scheduler:
                                    filename = f'{filename}_{curr_config.scheduler}'
                                    curr_config.project.name = f'{method}_b{block_size}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}_{curr_config.scheduler}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)
                                
            elif method == 'grad_fixed_new':
                for guidance_scale in [0.3]: # np.arange(100, 600, 100): # [25, 50, 100, 200, 500]:

                    for prompt_idx in range(num_prompts):
                        st = 0.6
                        et = 0.2

                        if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                            for target_idx in range(num_targets):

                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{int(float(round(guidance_scale,1))*10)}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.guidance_scale = float(guidance_scale)
                                curr_config.guidance.target_idxs = [target_idx]
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                
                                filename = f'{method}{int(float(round(guidance_scale,1))*10)}_p{prompt_idx}_t{target_idx}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)

                        else:
                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                            curr_config.project.promptspath = _SCORERS[scorer]
                            
                            curr_config.guidance.block_size = 5
                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.num_inference_steps = 100
                            curr_config.guidance.guidance_scale = float(guidance_scale)
                            curr_config.guidance.prompt_idxs = [prompt_idx]
                            del(curr_config.guidance["target_idxs"])
                            
                            curr_config.guidance.num_images_per_prompt = 5
                            curr_config.guidance.num_gen_target_images_per_prompt = 5
                            
                            curr_config.guidance.start_time = float(st)
                            curr_config.guidance.end_time = float(et)
                            
                            filename = f'{method}_p{prompt_idx}_st{int(st*10)}_et{int(et*10)}_{scorer}_gs{int(guidance_scale*10)}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)
                            
            elif method == 'grad_i2i' or method == 'grad_i2i_mpgd':
                for prompt_idx in range(num_prompts)[-1:]:

                    if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                        for target_idx in range(num_targets)[-1:]: 
                                for guidance_scale in [20,25,30]:
                                    for percent_noise in [1.0]:# np.arange(0.4, 1.0, 0.1):
                                        
                                        curr_config = copy.deepcopy(template)
                                        curr_config.project.name = f'{method}{guidance_scale}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                        curr_config.project.promptspath = _SCORERS[scorer]

                                        curr_config.guidance.method = method
                                        curr_config.guidance.scorer = scorer
                                        curr_config.guidance.guidance_scale = int(guidance_scale)
                                        curr_config.guidance.percent_noise = float(round(percent_noise,1))
                                        curr_config.guidance.target_idxs = [target_idx]
                                        curr_config.guidance.prompt_idxs = [prompt_idx]
                                        
                                        filename = f'{method}{int(float(round(guidance_scale,1))*10)}_r{int(float(round(percent_noise,1))*10)}_p{prompt_idx}_t{target_idx}_{scorer}'
                                        savepath = curr_path.joinpath(f'{filename}.yaml')
                                        OmegaConf.save(curr_config, savepath)
                        
                    else:
                        for guidance_scale in [7]:
                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}{guidance_scale}_{scorer}'
                            curr_config.project.promptspath = _SCORERS[scorer]
                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.guidance_scale = int(guidance_scale)
                            curr_config.guidance.prompt_idxs = [prompt_idx]
                            
                            filename = f'{method}{int(float(round(guidance_scale,1))*10)}_p{prompt_idx}_{scorer}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)

                
            elif method in ['c_code']:

                num_prompts = 3 if scorer == 'facedetector' else 4
                num_targets = 3

                for num_samples in [40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]

                        # pc = [0.6] if 'style' in scorer else [0.7]
                        for percent_noise in [0.8]:
                            # pt = [4,5] if 'style' in scorer else [3,4]
                            for prompt_idx in range(num_prompts):# pt 

                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    curr_config.project.name = f'{method}{num_samples}_b{block_size}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    curr_config.guidance.percent_noise = float(round(percent_noise,1))

                                    filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_r{float(round(percent_noise,1))}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

if __name__ == '__main__':
    create_function()