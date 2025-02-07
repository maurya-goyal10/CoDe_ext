import torch
import warnings
import logging
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.distributions as D

from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from src.model import MLP, Block
from src.utils import PositionalEmbedding, SinusoidalEmbedding, IdentityEmbedding
from src.noise_scheduler import NoiseScheduler

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

OUTPUTS_DIR = 'outputs'
DISABLE_TQDM = False

class Reward(nn.Module):

    def __init__(self, target_val1:int = 7, target_val2:int = 3, isgaussian: bool = True):
        super().__init__()

        self.isgaussian = isgaussian
        self.target1 = target_val1
        self.target2 = target_val2

    def forward(self, x, num_samples=None):
        
        isnumpy = False
        if type(x) is np.ndarray:
            isnumpy = True
            x = torch.tensor(x).unsqueeze(0)
        
        num_samples = x.shape[0]

        with torch.no_grad():

            if self.isgaussian:
                curr_target = torch.tensor([[self.target1, self.target1]]).long().to(x.device)
                curr_target = curr_target.repeat(num_samples, 1)

            else:
                curr_target = torch.tensor([[self.target1, self.target1]]).long()
                curr_target = curr_target.repeat(num_samples, 1)

                curr_target[:,0] = torch.tensor([self.target2 if sample < self.target1 else self.target1 for sample in x[:,1].tolist()]) 
                curr_target = curr_target.long().to(x.device)                

        reward = torch.exp(-(curr_target - x).pow(2).sum(1).sqrt())

        if isnumpy:
            return reward.item()
        else:
            return reward

def tokenlookahead(model, init_noise, noise_scheduler, isgaussian):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'tlg_gradx0')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
    reward_fn = Reward(target_val1=7, target_val2=3, isgaussian=isgaussian)

    # Function to compute gradient wrt x
    def cond_fn(x):
        grads = []
        losses = []
        for idx in range(len(x)):
            with torch.enable_grad():
                x_in = x[idx].detach().unsqueeze(0).requires_grad_(True)
                out = torch.log(reward_fn(x_in))
                grads.append(torch.autograd.grad(out, x_in)[0])
                losses.append(out)

        losses.append(out)
        grads = torch.cat(grads, dim=0)
        losses = torch.cat(losses, dim=0)
        return losses, grads
    

    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    for gscale in guidance_scale:

        torch.cuda.empty_cache()

        logger.info(f'Token-based Look ahead lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}{"_ng" if not isgaussian else ""}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        sample = copy.deepcopy(init_noise)

        int_samples_pi = []
        int_samples_p = []

        # div = 0
        timesteps = list(range(len(noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual = model(sample, t)

            pred_x0 = noise_scheduler.reconstruct_x0(sample, t[0], residual)
            _, grad = cond_fn(pred_x0)

            curr_sample = copy.deepcopy(sample)

            sample = noise_scheduler.step_wgrad(residual, t[0], sample, grad, scale=gscale)
            int_samples_pi.append(sample.detach().cpu())

            p_samples = []
            for _ in range(5):
                sample_temp = copy.deepcopy(curr_sample)
                sample_temp = noise_scheduler.step(residual, t[0], sample_temp)
                p_samples.append(sample_temp.detach().cpu())

            int_samples_p.append(p_samples) 

            del p_samples, curr_sample, sample_temp

            # if i == 0:
            #     div = losses
            # else:
            #     div += losses

        sample = sample.detach().cpu()
        # div = div.detach().cpu()

        torch.save(sample, f'{curr_path}/gensamples.pt')
        torch.save(int_samples_pi, f'{curr_path}/int_samples_pi.pt')
        torch.save(int_samples_p, f'{curr_path}/int_samples_p.pt')

        # Plot 
        data_df = pd.DataFrame({'x1': sample[:, 0], 'x2': sample[:, 1]})
        
        fig, ax = plt.subplots()
        sns.kdeplot(data_df, x='x1', y='x2', fill=True, ax=ax, cmap="Blues")
        ax.scatter(sample[:, 0], sample[:, 1], c='b', alpha=0.5, s=10)
        ax.axis([0, 10 , 0, 10])
        plt.grid()

        plt.savefig(f'{curr_path}/dist.png',dpi=300)
        plt.close()

def blocklookahead(model, init_noise, noise_scheduler, isgaussian):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'blg')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
    reward_fn = Reward(target_val1=7, target_val2=3, isgaussian=isgaussian)

    # Function to compute gradient wrt x
    def cond_fn(x):
        grads = []
        losses = []
        for idx in range(len(x)):
            with torch.enable_grad():
                x_in = x[idx].detach().unsqueeze(0).requires_grad_(True)
                out = torch.log(reward_fn(x_in))
                grads.append(torch.autograd.grad(out, x_in)[0])
                losses.append(out)

        losses.append(out)
        grads = torch.cat(grads, dim=0)
        losses = torch.cat(losses, dim=0)
        return losses, grads
    
    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    block_size = np.arange(2, 21, 2).tolist()
    block_size.extend(np.arange(20, 55, 5).tolist())

    for gscale in guidance_scale:

        logger.info(f'Block-based Look ahead lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}{"_ng" if not isgaussian else ""}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        for bsize in block_size:

            logger.info(f'Block size {str(bsize)}')

            torch.cuda.empty_cache()
            
            # Export directory
            curr_path_b = curr_path.joinpath(f'block_{str(int(bsize))}')
            if not Path.exists(curr_path_b):
                Path.mkdir(curr_path_b, exist_ok=True, parents=True)

            div = 0
            div_init = False
            sample = copy.deepcopy(init_noise)

            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, num_samples)).long()
                with torch.no_grad():
                    residual = model(sample, t)

                if t[0] % bsize == 0:
                    pred_x0 = noise_scheduler.reconstruct_x0(sample, t[0], residual)
                    losses, grad = cond_fn(pred_x0)

                    sample = noise_scheduler.step_wgrad(residual, t[0], sample, grad, scale=gscale)

                    if not div_init:
                        div = losses
                        div_init = True
                    else:
                        div += losses
                else:
                    sample = noise_scheduler.step(residual, t[0], sample)

            sample = sample.detach().cpu()
            div = div.detach().cpu()

            torch.save(sample, f'{curr_path_b}/gensamples.pt')
            torch.save(div, f'{curr_path_b}/genlosses.pt')

            # Plot 
            data_df = pd.DataFrame({'x1': sample[:, 0], 'x2': sample[:, 1]})

            fig, ax = plt.subplots()
            sns.kdeplot(data_df, x='x1', y='x2', fill=True, ax=ax, cmap="Blues")
            ax.scatter(sample[:, 0], sample[:, 1], c='b', alpha=0.5, s=10)
            ax.axis([0, 10 , 0, 10])
            plt.grid()

            plt.savefig(f'{curr_path_b}/dist.png',dpi=300)
            plt.close()

def bestofk(model, init_noise, noise_scheduler, isgaussian):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'bok')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
    reward_fn = Reward(target_val1=7, target_val2=3, isgaussian=isgaussian)

    sample = copy.deepcopy(init_noise)
    num_samples = sample.shape[0]

    n_vals = np.arange(2, 20, 2).tolist()
    n_vals.extend(np.arange(20, 110, 10).tolist())

    for n in n_vals:

        result = []
        result_uncond = []

        # Export directory
        curr_path = export_path.joinpath(f'n{str(int(n))}{"_ng" if not isgaussian else ""}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        # Generate samples
        for idx in tqdm(range(num_samples), total=num_samples):

            curr_sample = copy.deepcopy(sample[idx]).unsqueeze(0)
            curr_sample = curr_sample.repeat(n, 1)

            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(timesteps):
                t = torch.from_numpy(np.repeat(t, n)).long()
                with torch.no_grad():
                    residual = model(curr_sample, t)

                curr_sample = noise_scheduler.step(residual, t[0], curr_sample)
                
            reward = reward_fn(curr_sample, num_samples=n)
            select_ind = torch.max(reward, dim=0)[1]
            result.append(curr_sample[select_ind.item()].unsqueeze(0))
            result_uncond.append(curr_sample)

        result = torch.cat(result, dim=0).detach().cpu()
        result_uncond = torch.cat(result_uncond, dim=0).detach().cpu()

        torch.save(result, f'{curr_path}/gensamples.pt')
        torch.save(result_uncond, f'{curr_path}/gensamples_uncond.pt')

        # Plot 
        data_df = pd.DataFrame({'x1': result[:, 0], 'x2': result[:, 1]})

        fig, ax = plt.subplots()
        sns.kdeplot(data_df, x='x1', y='x2', fill=True, ax=ax, cmap="Blues")
        ax.scatter(result[:, 0], result[:, 1], c='b', alpha=0.5, s=10)
        ax.axis([0, 10 , 0, 10])
        plt.grid()

        plt.savefig(f'{curr_path}/dist.png',dpi=300)
        plt.close()

def bestofk_int(model, init_noise, noise_scheduler, isgaussian, blocksize=1):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'bok_b{blocksize}')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
    reward_fn = Reward(target_val1=7, target_val2=3, isgaussian=isgaussian)

    num_samples = init_noise.shape[0]

    n_vals = np.arange(1, 20, 2).tolist()
    n_vals.extend(np.arange(20, 110, 10).tolist())

    for n in n_vals:

        # Export directory
        curr_path = export_path.joinpath(f'n{str(int(n))}{"_ng" if not isgaussian else ""}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        sample = copy.deepcopy(init_noise)
        
        # Generate samples
        timesteps = list(range(len(noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):

            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual = model(sample, t)

            if (t[0] > 0) and (t[0] % blocksize == 0):
                gen_reward = []
                gen_sample = []
                prev_t = torch.from_numpy(np.repeat(timesteps[i + 1], num_samples)).long()

                for _ in range(n):
                    sample_temp = copy.deepcopy(sample)

                    # Sampling possible next steps x_t-1
                    sample_temp = noise_scheduler.step(residual, t[0], sample_temp)

                    # Estimate the clean sample from noisy sample
                    with torch.no_grad():
                        residual_temp = model(sample_temp, prev_t)

                    pred_x0 = noise_scheduler.reconstruct_x0(sample_temp, prev_t[0], residual_temp)

                    reward = reward_fn(pred_x0, num_samples=num_samples)
                    
                    gen_reward.append(reward.unsqueeze(0))
                    gen_sample.append(copy.deepcopy(sample_temp.unsqueeze(0)))

                # Find the direction that minimizes the loss
                select_ind = torch.max(torch.cat(gen_reward), dim=0)[1]
                gen_sample = torch.cat(gen_sample, dim=0)
                gen_sample = gen_sample.permute(1,0,2)
                sample = torch.cat([x[select_ind[idx]].unsqueeze(0) for idx, x in enumerate(gen_sample)], dim=0) # TODO: Make it efficient
            else:
                sample = noise_scheduler.step(residual, t[0], sample)

        sample = sample.detach().cpu()

        torch.save(sample, f'{curr_path}/gensamples.pt')

        # Plot 
        data_df = pd.DataFrame({'x1': sample[:, 0], 'x2': sample[:, 1]})

        fig, ax = plt.subplots()
        sns.kdeplot(data_df, x='x1', y='x2', fill=True, ax=ax, cmap="Blues")
        ax.scatter(sample[:, 0], sample[:, 1], c='b', alpha=0.5, s=10)
        ax.axis([0, 10 , 0, 10])
        plt.grid()

        plt.savefig(f'{curr_path}/dist.png',dpi=300)
        plt.close()

def tokenlookahead2(model, init_noise, noise_scheduler, isgaussian):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'tlg_gradxt')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
    reward_fn = Reward(target_val1=7, target_val2=3, isgaussian=isgaussian)

    # Function to compute gradient wrt x
    def cond_fn(x, t, res):
        grads = []
        losses = []
        for idx in range(len(x)):
            with torch.enable_grad():
                x_in = x[idx].detach().unsqueeze(0).requires_grad_(True)
                out = noise_scheduler.reconstruct_x0(x_in, t, res[idx])
                out = torch.log(reward_fn(x_in))
                grads.append(torch.autograd.grad(out, x_in)[0])
                losses.append(out)

        losses.append(out)
        grads = torch.cat(grads, dim=0)
        losses = torch.cat(losses, dim=0)
        return losses, grads
    

    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    for gscale in guidance_scale:

        torch.cuda.empty_cache()

        logger.info(f'Token-based Look ahead lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}{"_ng" if not isgaussian else ""}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        sample = copy.deepcopy(init_noise)

        # div = 0
        int_samples_pi = []
        int_samples_p = []
        timesteps = list(range(len(noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual = model(sample, t)

            _, grad = cond_fn(sample, t[0], residual)

            curr_sample = copy.deepcopy(sample)

            sample = noise_scheduler.step_wgrad(residual, t[0], sample, grad, scale=gscale)
            int_samples_pi.append(sample.detach().cpu())

            p_samples = []
            for _ in range(5):
                sample_temp = copy.deepcopy(curr_sample)
                sample_temp = noise_scheduler.step(residual, t[0], sample_temp)
                p_samples.append(sample_temp.detach().cpu())

            int_samples_p.append(p_samples) 

            del p_samples, curr_sample, sample_temp

            # if i == 0:
            #     div = losses
            # else:
            #     div += losses

        sample = sample.detach().cpu()
        # div = div.detach().cpu()

        torch.save(sample, f'{curr_path}/gensamples.pt')
        # torch.save(div, f'{curr_path}/genlosses.pt')
        torch.save(int_samples_pi, f'{curr_path}/int_samples_pi.pt')
        torch.save(int_samples_p, f'{curr_path}/int_samples_p.pt')

        # Plot 
        data_df = pd.DataFrame({'x1': sample[:, 0], 'x2': sample[:, 1]})
        
        fig, ax = plt.subplots()
        sns.kdeplot(data_df, x='x1', y='x2', fill=True, ax=ax, cmap="Blues")
        ax.scatter(sample[:, 0], sample[:, 1], c='b', alpha=0.5, s=10)
        ax.axis([0, 10 , 0, 10])
        plt.grid()

        plt.savefig(f'{curr_path}/dist.png',dpi=300)
        plt.close()

def uniguide(model, init_noise, noise_scheduler, isgaussian):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'ug2')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    if isgaussian:
        curr_target = torch.tensor([[7, 7]]).long().to(init_noise.device)
    
    reward_fn = Reward(target_val1=7, target_val2=3, isgaussian=isgaussian)

    # Function to compute gradient wrt x
    # def cond_fn(x, res, t):
    #     grads = []
    #     losses = []
    #     for idx in range(len(x)):
    #         with torch.enable_grad():
    #             x_in = x[idx].detach().unsqueeze(0).requires_grad_(True)
    #             out = noise_scheduler.reconstruct_x0(x_in, t, res[idx])
    #             out = torch.log(reward_fn(out))
    #             grads.append(torch.autograd.grad(out, x_in)[0])
    #             losses.append(out)

    #     losses.append(out)
    #     grads = torch.cat(grads, dim=0)
    #     losses = torch.cat(losses, dim=0)
    #     return losses, grads
    
    def cond_fn(x):
        grads = []
        losses = []
        for idx in range(len(x)):
            with torch.enable_grad():
                x_in = x[idx].detach().unsqueeze(0).requires_grad_(True)
                out = torch.log(reward_fn(x_in))
                grads.append(torch.autograd.grad(out, x_in)[0])
                losses.append(out)

        losses.append(out)
        grads = torch.cat(grads, dim=0)
        losses = torch.cat(losses, dim=0)
        return losses, grads
    
    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    # guidance_scale = np.arange(0.7, 2, 0.1).tolist()
    # guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())
    

    # guidance_scale = [1,2,3,5,6]

    # Loop over refinement steps
    # refine_steps = np.arange(1,20,1).tolist()

    refine_steps = [2,3]

    for gscale in guidance_scale:

        logger.info(f'Universal guidance lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}{"_ng" if not isgaussian else ""}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        for rsteps in refine_steps:

            # Export directory
            curr_path_r = curr_path.joinpath(f'rsteps_{str(int(rsteps))}')
            if not Path.exists(curr_path_r):
                Path.mkdir(curr_path_r, exist_ok=True, parents=True)
        
            torch.cuda.empty_cache()

            sample = copy.deepcopy(init_noise)

            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, num_samples)).long()


                for k in range(rsteps):

                    with torch.no_grad():
                        residual = model(sample, t)

                    residual_updated = copy.deepcopy(residual)

                    # forward guidance  
                    pred_x0 = noise_scheduler.reconstruct_x0(sample, t[0], residual)

                    losses, grad = cond_fn(pred_x0)
                    residual_updated = residual_updated - (gscale * noise_scheduler.sqrt_one_minus_alphas_cumprod[t[0]] * grad.float())
                
                    if True: # SGD

                        deltas = - 0.01 * (curr_target - pred_x0).pow(2).sqrt()

                        # backward guidance  
                        residual_updated = residual_updated - (noise_scheduler.sqrt_alphas_cumprod[t[0]] / noise_scheduler.sqrt_one_minus_alphas_cumprod[t[0]]) * deltas
                
                    sample = noise_scheduler.step(residual_updated, t[0], sample)

                if k < (rsteps - 1):
                    noise = torch.randn_like(sample).to(sample.device)
                    sample = (noise_scheduler.sqrt_alphas[t[0]]) * sample + (noise_scheduler.sqrt_betas[t[0]]) * noise

            sample = sample.detach().cpu()

            torch.save(sample, f'{curr_path_r}/gensamples.pt')

            # Plot 
            data_df = pd.DataFrame({'x1': sample[:, 0], 'x2': sample[:, 1]})
            
            fig, ax = plt.subplots()
            sns.kdeplot(data_df, x='x1', y='x2', fill=True, ax=ax, cmap="Blues")
            ax.scatter(sample[:, 0], sample[:, 1], c='b', alpha=0.5, s=10)
            ax.axis([0, 10 , 0, 10])
            plt.grid()

            plt.savefig(f'{curr_path_r}/dist.png',dpi=300)
            plt.close()

def compute_stepval(samples_pi, samples_p, timestep, scale, model, noise_scheduler):

    scale = 5
    IS_GAUSSIAN_REWARD = True
    reward_fn = Reward(target_val1=7, target_val2=3, isgaussian=IS_GAUSSIAN_REWARD)

    breakpoint()
    # compute lambda * V(x_{t-1})
    samples_pi = samples_pi.to('cuda')
    t = torch.from_numpy(np.repeat(timestep, samples_pi.shape[0])).long()
    with torch.no_grad():
        residual_pi = model(samples_pi, t)

    pred_orig_pi = noise_scheduler.reconstruct_x0(samples_pi, t[0], residual_pi)
    rewards_pi = scale * reward_fn(pred_orig_pi)

    # compute elements for Z_{lambda}
    samples_p = torch.cat([x.unsqueeze(0) for x in samples_p])
    samples_p = samples_p.to('cuda')
    n, num_samples, dims = samples_p.shape

    samples_p = samples_p.reshape(-1,2)
    t = torch.from_numpy(np.repeat(timestep, samples_p.shape[0])).long()
    with torch.no_grad():
        residual_p = model(samples_p, t)

    pred_orig_p = noise_scheduler.reconstruct_x0(samples_p, t[0], residual_p)
    rewards_p = torch.exp(scale * reward_fn(pred_orig_p))
    rewards_p = rewards_p.reshape(n, num_samples)

    mu_theta_p = noise_scheduler.q_posterior(pred_orig_p, samples_p, t[0])
    mu_theta_p = mu_theta_p.reshape(n, num_samples, dims)

    beta_theta_p = (1.0 - noise_scheduler.alphas_cumprod_prev)/(1.0 - noise_scheduler.alphas_cumprod)*noise_scheduler.betas

    samples_p = samples_p.reshape(n, num_samples, dims)

    result = []
    for idx in range(num_samples):

        # compute Z_{lambda}
        partition_val = []
        for idx_n in range(n):
            dist = D.MultivariateNormal(mu_theta_p[idx_n][idx], (beta_theta_p[t[0]] * torch.eye(2)).to('cuda'))
            p = torch.exp(dist.log_prob(samples_p[idx_n][idx]))
            partition_val.append(p*rewards_p[idx_n][idx])

        breakpoint()
        mean_part = torch.mean(torch.tensor(partition_val))
        partition_val = torch.log(mean_part)

        # compute lambda * V(x_{t-1}) - Z_{lambda}
        result.append(rewards_pi[idx] - partition_val)

    return torch.tensor(result)



def main():

    log_path = Path('logs')

    # Set up logger
    if not Path.exists(log_path):
        Path.mkdir(log_path, exist_ok=True, parents=True)

    log_path = log_path.joinpath('genplots_log.txt')

    logging.basicConfig(level = logging.INFO,
                        filemode = 'w',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename = log_path)
    logger = logging.getLogger()

    logger.info('Log file is %s.' % (log_path))

    # Define beta schedule
    T = 1000
    noise_scheduler = NoiseScheduler(
        num_timesteps=T,
        beta_schedule="linear")
    
    # Load trained model
    model = torch.load(f'{OUTPUTS_DIR}/model.pt')
    model = model.to('cuda')
    model.eval()

    # Initial noise
    samples = torch.load(f'{OUTPUTS_DIR}/noise.pt').to('cuda')

    # Generate plots
    tokenlookahead(model, samples, noise_scheduler, isgaussian=True)
    tokenlookahead(model, samples, noise_scheduler, isgaussian=False)

    # tokenlookahead2(model, samples, noise_scheduler, isgaussian=True)
    # tokenlookahead2(model, samples, noise_scheduler, isgaussian=False)

    # uniguide(model, samples, noise_scheduler, isgaussian=True)

    # blocklookahead(model, samples, noise_scheduler, isgaussian=True)
    # blocklookahead(model, samples, noise_scheduler, isgaussian=False)

    # bestofk(model, samples, noise_scheduler, isgaussian=True)
    # bestofk(model, samples, noise_scheduler, isgaussian=False)

    # block_size = np.arange(2, 21, 2).tolist()
    # block_size.extend(np.arange(20, 55, 5).tolist())

    block_size = [50, 100, 500]

    for bs in block_size:
        bestofk_int(model, samples, noise_scheduler, isgaussian=True, blocksize=bs)
        bestofk_int(model, samples, noise_scheduler, isgaussian=False, blocksize=bs)

    # scale = 5
    # IS_GAUSSIAN_REWARD = True

    # int_samples_p = torch.load(f'{OUTPUTS_DIR}/tlg_gradx0/lambda_{int(scale*10)}{"_ng" if not IS_GAUSSIAN_REWARD else ""}/int_samples_p.pt')
    # int_samples_pi = torch.load(f'{OUTPUTS_DIR}/tlg_gradx0/lambda_{int(scale*10)}{"_ng" if not IS_GAUSSIAN_REWARD else ""}/int_samples_pi.pt')
    
    # div = None

    # timesteps = list(range(len(noise_scheduler)))[::-1]
    # for i, t_loop in enumerate(tqdm(timesteps)):

    #     curr_samples_pi = int_samples_pi[i]
    #     curr_samples_p = int_samples_p[i]

    #     try:
    #         res = compute_stepval(curr_samples_pi, curr_samples_p, t_loop, scale=scale, model=model, noise_scheduler=noise_scheduler)
    #     except:
    #         print(f'{i} {t_loop}')
    #         break

    #     if i == 0:
    #         div = res
    #     else:
    #         div += res



if __name__ == '__main__':
    main()