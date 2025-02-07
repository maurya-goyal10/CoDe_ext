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
from src.model import MLP, Block, Conditional_MLP
from src.utils import PositionalEmbedding, SinusoidalEmbedding, IdentityEmbedding
from src.noise_scheduler import NoiseScheduler

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

OUTPUTS_DIR = 'outputs_gmm'
DISABLE_TQDM = False

class GaussianMixture:
  
    def __init__(self, mus, covs, weights, device='cuda'):
        """
        mus: a list of K 1d np arrays (D,)
        covs: a list of K 2d np arrays (D, D)
        weights: a list or array of K unnormalized non-negative weights, signifying the possibility of sampling from each branch.
        They will be normalized to sum to 1. If they sum to zero, it will err.
        """
        self.n_component = len(mus)
        self.mus = mus
        self.covs = covs
        self.device = device
        # self.precs = [np.linalg.inv(cov) for cov in covs]
        self.weights = np.array(weights)
        self.norm_weights = self.weights / self.weights.sum()
        self.RVs = []
        for i in range(len(mus)):
            self.RVs.append(D.Independent(D.Normal(mus[i], covs[i]), 1))
        self.dim = len(mus[0])

    def add_component(self, mu, cov, weight=1):
        self.mus.append(mu)
        self.covs.append(cov)
        self.precs.append(np.linalg.inv(cov.cpu().numpy()))
        self.RVs.append(D.Independent(D.Normal(mu, cov), 1))
        self.weights.append(weight)
        self.norm_weights = self.weights / self.weights.sum()
        self.n_component += 1

    def pdf_decompose(self, x):
        """
        probability density (PDF) at $x$.
        """
        component_pdf = []
        prob = None
        for weight, RV in zip(self.norm_weights, self.RVs):
            pdf = weight * RV.log_prob(x).exp()
            prob = pdf if prob is None else (prob + pdf)
            component_pdf.append(pdf)
        component_pdf = np.array(component_pdf)
        return prob, component_pdf

    def pdf(self, x):
        """
        probability density (PDF) at $x$.
        """
        isnumpy = False
        if type(x) is np.ndarray:
            isnumpy = True
            x = torch.from_numpy(x).to(self.mus[0].device)

        prob = None
        for weight, RV in zip(self.norm_weights, self.RVs):
            pdf = weight * RV.log_prob(x).exp()
            prob = pdf if prob is None else (prob + pdf)

        if isnumpy:
            prob = prob.cpu().numpy()
        
        return prob

    def score(self, x):
        """
        Compute the score $\nabla_x \log p(x)$ for the given $x$.
        """
        isnumpy = False
        if type(x) is np.ndarray:
            isnumpy = True
            x = torch.from_numpy(x).to(torch.float).to(self.mus[0].device)

        component_pdf = np.array([rv.log_prob(x).exp().cpu().numpy() for rv in self.RVs])
        component_pdf = torch.from_numpy(component_pdf).T
        
        # if isnumpy:
        #   component_pdf = np.array([rv.log_prob(x).exp().cpu().numpy() for rv in self.RVs]).T
        # else:
        #   component_pdf = np.array([rv.log_prob(x).exp() for rv in self.RVs]).T
        
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)
        participance = participance.to(self.mus[0].device)

        scores = torch.zeros_like(x)
        for i in range(self.n_component):
            gradvec = (- (x - self.mus[i]) @ torch.diag(self.covs[i]))
            scores += participance[:, i:i+1] * gradvec

        if isnumpy:
            scores = scores.cpu().numpy()

        return scores

    def score_decompose(self, x):
        """
        Compute the grad to each branch for the score $\nabla_x \log p(x)$ for the given $x$.
        """
        component_pdf = np.array([rv.log_prob(x).exp() for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / weighted_compon_pdf.sum(axis=1, keepdims=True)

        gradvec_list = []
        for i in range(self.n_component):
            gradvec = - (x - self.mus[i]) @ torch.diag(self.covs[i])
            gradvec_list.append(gradvec)
        # scores += participance[:, i:i+1] * gradvec

        return gradvec_list, participance

    def sample(self, N):
        """ Draw N samples from Gaussian mixture
        Procedure:
        Draw N samples from each Gaussian
        Draw N indices, according to the weights.
        Choose sample between the branches according to the indices.
        """
        rand_component = np.random.choice(self.n_component, size=N, p=self.norm_weights)
        all_samples = np.array([rv.sample((N,)).cpu().numpy() for rv in self.RVs])
        gmm_samps = all_samples[rand_component, np.arange(N),:]
        return gmm_samps, rand_component, all_samples

def tokenlookahead(model, init_noise, noise_scheduler, gmm):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'tlg_gradx0')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    for gscale in guidance_scale:

        torch.cuda.empty_cache()

        logger.info(f'Token-based Look ahead lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}')
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
            grad = gmm.score(pred_x0)

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

def blocklookahead(model, init_noise, noise_scheduler, gmm):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'blg')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    block_size = np.arange(2, 21, 2).tolist()
    block_size.extend(np.arange(20, 55, 5).tolist())

    for gscale in guidance_scale:

        logger.info(f'Block-based Look ahead lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        for bsize in block_size:

            logger.info(f'Block size {str(bsize)}')

            torch.cuda.empty_cache()
            
            # Export directory
            curr_path_b = curr_path.joinpath(f'block_{str(int(bsize))}')
            if not Path.exists(curr_path_b):
                Path.mkdir(curr_path_b, exist_ok=True, parents=True)

            sample = copy.deepcopy(init_noise)

            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, num_samples)).long()
                with torch.no_grad():
                    residual = model(sample, t)

                if t[0] % bsize == 0:
                    pred_x0 = noise_scheduler.reconstruct_x0(sample, t[0], residual)
                    grad = gmm.score(pred_x0)

                    sample = noise_scheduler.step_wgrad(residual, t[0], sample, grad, scale=gscale)
                else:
                    sample = noise_scheduler.step(residual, t[0], sample)

            sample = sample.detach().cpu()

            torch.save(sample, f'{curr_path_b}/gensamples.pt')

            # Plot 
            data_df = pd.DataFrame({'x1': sample[:, 0], 'x2': sample[:, 1]})

            fig, ax = plt.subplots()
            sns.kdeplot(data_df, x='x1', y='x2', fill=True, ax=ax, cmap="Blues")
            ax.scatter(sample[:, 0], sample[:, 1], c='b', alpha=0.5, s=10)
            ax.axis([0, 10 , 0, 10])
            plt.grid()

            plt.savefig(f'{curr_path_b}/dist.png',dpi=300)
            plt.close()

def bestofk(model, init_noise, noise_scheduler, gmm):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'bok')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    sample = copy.deepcopy(init_noise)
    num_samples = sample.shape[0]

    n_vals = np.arange(2, 20, 2).tolist()
    n_vals.extend(np.arange(20, 110, 10).tolist())

    for n in n_vals:

        result = []
        result_uncond = []

        # Export directory
        curr_path = export_path.joinpath(f'n{str(int(n))}')
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
                
            reward = gmm.pdf(curr_sample)
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

def bestofk_int(model, init_noise, noise_scheduler, gmm, blocksize=1):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'bok_b{blocksize}')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    num_samples = init_noise.shape[0]

    n_vals = np.arange(1, 20, 2).tolist()
    n_vals.extend(np.arange(20, 110, 10).tolist())

    for n in n_vals:

        # Export directory
        curr_path = export_path.joinpath(f'n{str(int(n))}')
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

                    reward = gmm.pdf(pred_x0)
                    
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

def tokenlookahead2(model, init_noise, noise_scheduler, gmm):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'tlg_gradxt')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    for gscale in guidance_scale:

        torch.cuda.empty_cache()

        logger.info(f'Token-based Look ahead lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}')
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

            grad = gmm.score(sample)

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

def uniguide(model, init_noise, noise_scheduler, gmm):

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'ug2')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)
    
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
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}')
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

                    grad = gmm.score(pred_x0)
                    residual_updated = residual_updated - (gscale * noise_scheduler.sqrt_one_minus_alphas_cumprod[t[0]] * grad.float())
                
                    if True: # SGD
                        
                        param = copy.deepcopy(pred_x0)
                        for i in range(5):
                            grad_sgd = gmm.score(param)
                            param = param - (0.01 * grad_sgd)
                        
                        deltas = param - pred_x0

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

def cfg(init_noise, noise_scheduler):

    # Load conditional model
    model_cond = torch.load(f'{OUTPUTS_DIR}/model_cond.pt')
    model_cond = model_cond.to('cuda')
    model_cond.eval()

    # Load unconditional model
    model_uncond = torch.load(f'{OUTPUTS_DIR}/model_uncond.pt')
    model_uncond = model_uncond.to('cuda')
    model_uncond.eval()

    # Export path
    export_path = Path(OUTPUTS_DIR).joinpath(f'cfg')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True, parents=True)

    num_samples = init_noise.shape[0]

    # Loop over guidance scale
    guidance_scale = np.arange(0.1, 2, 0.1).tolist()
    guidance_scale.extend(np.arange(2, 5.5, 0.5).tolist())

    for gscale in guidance_scale:

        torch.cuda.empty_cache()

        logger.info(f'Classifier-free guidance lambda {str(gscale)}')

        # Export directory
        curr_path = export_path.joinpath(f'lambda_{str(int(gscale*10))}')
        if not Path.exists(curr_path):
            Path.mkdir(curr_path, exist_ok=True, parents=True)

        sample = copy.deepcopy(init_noise)
        
        per_class = len(sample)//3
        class_label = torch.cat([torch.zeros_like(sample[:per_class, [0]]), 
                                 torch.ones_like(sample[per_class:per_class*2, [0]]), 
                                 torch.ones_like(sample[per_class*2:, [0]]) * 2])
        class_label = class_label.to(sample.device)

        timesteps = list(range(len(noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual_cond = model_cond(sample, class_label, t)
                residual_uncond = model_uncond(sample, t)

                residual = ((1 + gscale) * residual_cond) - (gscale * residual_uncond)

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

def main():

    log_path = Path('logs')

    # Set up logger
    if not Path.exists(log_path):
        Path.mkdir(log_path, exist_ok=True, parents=True)

    log_path = log_path.joinpath('genplots_log_cfg.txt')

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
    
    # Define gaussian mixture
    mu1 = torch.tensor([2.0, 8.0]).to('cuda')
    Cov1 = torch.tensor([1.0, 1.0]).to('cuda')

    mu2 = torch.tensor([5.0, 2.0]).to('cuda')
    Cov2 = torch.tensor([1.0, 1.0]).to('cuda')

    mu3 = torch.tensor([8.0, 8.0]).to('cuda')
    Cov3 = torch.tensor([1.0, 1.0]).to('cuda')

    gmm = GaussianMixture([mu1, mu2, mu3],[Cov1, Cov2, Cov3], [1.0, 1.0, 1.0])
    
    # Load trained model
    model = torch.load(f'{OUTPUTS_DIR}/model.pt')
    model = model.to('cuda')
    model.eval()

    # Initial noise
    samples = torch.load(f'{OUTPUTS_DIR}/noise.pt').to('cuda')

    # Generate plots

    cfg(samples, noise_scheduler)

    # tokenlookahead(model, samples, noise_scheduler, gmm=gmm)

    # tokenlookahead2(model, samples, noise_scheduler, gmm=gmm)

    # uniguide(model, samples, noise_scheduler, gmm=gmm)

    # blocklookahead(model, samples, noise_scheduler, gmm=gmm)

    # bestofk(model, samples, noise_scheduler, gmm=gmm)

    # block_size = np.arange(2, 21, 2).tolist()
    # block_size.extend(np.arange(20, 55, 5).tolist())

    # block_size.extend([50, 100, 500])

    # for bs in block_size:
    #     bestofk_int(model, samples, noise_scheduler, gmm=gmm, blocksize=bs)

if __name__ == '__main__':
    main()