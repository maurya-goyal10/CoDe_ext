import argparse, os, sys, glob
import cv2
import torch
import json
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from ldm.models.diffusion.aesthetic.aesthetic_scorer import AestheticScorer
from ldm.models.diffusion.pickscore.pickscore_scorer import PickScoreScorer
from ldm.models.diffusion.multireward.multi_reward import MultiReward

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from pathlib import Path

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def load_score(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

def save_score(file_path, score):
    with open(file_path, "w") as f:
        json.dump(score, f)

def update_score(file_path, prompt, new_score):
    results = load_score(file_path)
    if not isinstance(results, list):
        results = [results] 
    results.append(new_score)
    save_score(file_path, results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--style_ref_img_path",
        type=str,
        nargs="?",
        default="./style_images/xingkong.jpg",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.2,
        help="guidance scale like"
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/mpgd_pytorch/nonlinear/SD_style/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    # parser.add_argument(
    #     "--from_file",
    #     type=str,
    #     default="/tudelft.net/staff-umbrella/StudentsCVlab/mgoyal/CoDe_ext/AlignProp/assets/eval_simple_animals.txt",
    #     help="path to checkpoint of model",
    # )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--ref_img_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--alt_score",
        action='store_true',
        help="if enabled, uses alternate function for guidance",
    )
    parser.add_argument(
        "--weight1",
        type=int,
        default=1
    )
    parser.add_argument(
        "--weight2",
        type=int,
        default=1
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        if isinstance(opt.prompt, list):
            data = 1 * opt.prompt
        else:
            data = 1 * [opt.prompt]
        # prompt = opt.prompt
        # assert prompt is not None
        # data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, f"FreeDoM_multireward_rho{opt.rho}_aes{opt.weight1}_pic{opt.weight2}")
    # sample_path = os.path.join(outpath, f"FreeDoM_aesthetic_rho{opt.rho}")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    # with torch.no_grad():
    print(data)
    
    # image_encoder = AestheticScorer().cuda()
    image_encoder = MultiReward("aesthetic","pickscore",opt.weight1,opt.weight2).cuda()
    
    with precision_scope("cuda"):
        with model.ema_scope():

            for idx, prompt in enumerate(data):

                n = -1
                # for filename in tqdm(sorted(os.listdir(opt.style_ref_img_path))):
                #     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):

                #         n += 1
                #         num_images_per_prompt = opt.n_iter

                #         offset = 0
                #         savepath = Path(sample_path).joinpath(f'og_img_{n}').joinpath("images").joinpath(prompt[0])
                #         if Path.exists(savepath):

                #             images = [x for x in savepath.iterdir() if x.suffix == '.png']
                #             num_gen_images = len(images)
                #             if num_gen_images >= num_images_per_prompt:
                #                 print(f'Images found. Skipping prompt.')
                #                 continue

                #             elif num_gen_images < num_images_per_prompt:
                #                 offset = num_gen_images
                #                 num_images_per_prompt -= num_gen_images
                #                 print(f'Found {num_gen_images} images. Generating {num_images_per_prompt} more.')

                #         if not Path.exists(savepath):
                #             Path.mkdir(savepath, exist_ok=True, parents=True)

                #         target_img = Path(sample_path).joinpath(f'og_img_{n}').joinpath(f'og_img_{n}.png')
                #         if not Path.exists(target_img):
                #             img = Image.open(os.path.join(opt.style_ref_img_path, filename)).convert('RGB')
                #             img.save(target_img)

                #         for j in range(num_images_per_prompt):
                #             uc = None
                #             if opt.scale != 1.0:
                #                 uc = model.get_learned_conditioning(batch_size * [""])
                #             if isinstance(prompt, tuple):
                #                 prompt = list(prompt)
                #             c = model.get_learned_conditioning(prompt)
                #             shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                #             samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                #                                                 conditioning=c,
                #                                                 batch_size=opt.n_samples,
                #                                                 shape=shape,
                #                                                 verbose=False,
                #                                                 unconditional_guidance_scale=opt.scale,
                #                                                 unconditional_conditioning=uc,
                #                                                 eta=opt.ddim_eta,
                #                                                 x_T=start_code,
                #                                                 style_ref_img_path=os.path.join(opt.style_ref_img_path, filename))

                #             x_samples_ddim = model.decode_first_stage(samples_ddim)
                #             x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                #             x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).detach().numpy()

                #             # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                #             x_checked_image = x_samples_ddim

                #             x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                #             if not opt.skip_save:
                #                 for x_sample in x_checked_image_torch:
                #                     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                #                     img = Image.fromarray(x_sample.astype(np.uint8))
                #                     # img = put_watermark(img, wm_encoder)
                #                     img.save(os.path.join(savepath, f'{j + offset}.png'))
                #                     # base_count += 1


                n += 1
                num_images_per_prompt = opt.n_iter

                offset = 0
                savepath = Path(sample_path).joinpath("images").joinpath(f"{prompt}")
                # if Path.exists(savepath):

                #     images = [x for x in savepath.iterdir() if x.suffix == '.png']
                #     num_gen_images = len(images)
                #     if num_gen_images >= num_images_per_prompt:
                #         print(f'Images found. Skipping prompt.')
                #         continue

                #     elif num_gen_images < num_images_per_prompt:
                #         offset = num_gen_images
                #         num_images_per_prompt -= num_gen_images
                #         print(f'Found {num_gen_images} images. Generating {num_images_per_prompt} more.')

                if not Path.exists(savepath):
                    Path.mkdir(savepath, exist_ok=True, parents=True)

                # target_img = Path(sample_path).joinpath(f'og_img_{n}').joinpath(f'og_img_{n}.png')
                # if not Path.exists(target_img):
                #     img = Image.open(os.path.join(opt.style_ref_img_path, filename)).convert('RGB')
                #     img.save(target_img)

                for j in range(num_images_per_prompt):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    c = model.get_learned_conditioning(batch_size * [prompt])
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=batch_size,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        rho=opt.rho,
                                                        prompt = prompt,
                                                        image_encoder = image_encoder)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    # score = AestheticScorer().cuda().score(x_samples_ddim)[0].item()
                    score,score1,score2 = image_encoder.score(x_samples_ddim.detach(),prompt,return_all=True)
                    score = score[0].item()
                    score1 = score1[0].item()
                    score2 = score2[0].item()
                    # score = image_encoder.score(x_samples_ddim.detach())[0].item()
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).detach().numpy()

                    # print(f"The value of the aesthetic score is {aesthetic_score}")
                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                    x_checked_image = x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                    
                    score_dir = os.path.join(savepath, f"rewards.json")
                    score_dir1 = os.path.join(savepath, f"rewards1.json")
                    score_dir2 = os.path.join(savepath, f"rewards2.json")
                    
                    z = 0
                    if os.path.exists(score_dir):
                        with open(score_dir,"r") as f:
                            data = json.load(f)
                    if os.path.exists(score_dir1):
                        with open(score_dir1,"r") as f:
                            data1 = json.load(f)
                    if os.path.exists(score_dir2):
                        with open(score_dir2,"r") as f:
                            data2 = json.load(f)
                            
                        if data is not None:
                            z = len(data)

                    if not opt.skip_save:
                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(savepath, f'{z}.png'))
                            z += 1
                            # base_count += 1
                            update_score(score_dir,prompt,score) 
                            update_score(score_dir1,prompt,score1) 
                            update_score(score_dir2,prompt,score2) 
                # torch.cuda.empty_cache()

            # tic = time.time()
            # all_samples = list()
            # for n in trange(opt.n_iter, desc="Sampling"):
            #     for prompts in tqdm(data, desc="data"):
            #         uc = None
            #         if opt.scale != 1.0:
            #             uc = model.get_learned_conditioning(batch_size * [""])
            #         if isinstance(prompts, tuple):
            #             prompts = list(prompts)
            #         c = model.get_learned_conditioning(prompts)
            #         shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            #         samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
            #                                             conditioning=c,
            #                                             batch_size=opt.n_samples,
            #                                             shape=shape,
            #                                             verbose=False,
            #                                             unconditional_guidance_scale=opt.scale,
            #                                             unconditional_conditioning=uc,
            #                                             eta=opt.ddim_eta,
            #                                             x_T=start_code,
            #                                             style_ref_img_path=opt.style_ref_img_path)

            #         x_samples_ddim = model.decode_first_stage(samples_ddim)
            #         x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            #         x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).detach().numpy()

            #         # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
            #         x_checked_image = x_samples_ddim

            #         x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

            #         if not opt.skip_save:
            #             for x_sample in x_checked_image_torch:
            #                 x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            #                 img = Image.fromarray(x_sample.astype(np.uint8))
            #                 # img = put_watermark(img, wm_encoder)
            #                 img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            #                 base_count += 1

            #         if not opt.skip_grid:
            #             all_samples.append(x_checked_image_torch)

            # if not opt.skip_grid:
            #     # additionally, save as grid
            #     grid = torch.stack(all_samples, 0)
            #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            #     grid = make_grid(grid, nrow=n_rows)

            #     # to image
            #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            #     img = Image.fromarray(grid.astype(np.uint8))
            #     # img = put_watermark(img, wm_encoder)
            #     img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            #     grid_count += 1

            # toc = time.time()



if __name__ == "__main__":
    main()
