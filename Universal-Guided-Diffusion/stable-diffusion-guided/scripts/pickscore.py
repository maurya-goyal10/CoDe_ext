import argparse, os, sys, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import json
import torch
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

from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.optim import SGD, Adam, AdamW
import PIL
from torch.utils import data
from pathlib import Path
from PIL import Image
from torchvision import transforms, utils
import random
from helper import OptimizerDetails
import clip
import os
import inspect
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder

from transformers import AutoModel, CLIPProcessor

ASSETS_PATH = Path("../../BoN/assets/")

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

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, data_aug=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        random.shuffle(self.paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size), resample=PIL.Image.LANCZOS)

        return self.transform(img)

def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img

def cycle(dl):
    while True:
        for data in dl:
            yield data

import os
import errno
def create_folder(path):
    path = Path(path)
    if not Path.exists(path):
        Path.mkdir(path, exist_ok=True, parents=True)
    # try:
    #     os.mkdir(path)
    # except OSError as exc:
    #     if exc.errno != errno.EEXIST:
    #         raise
    #     pass

class PickScore(nn.Module):
    def __init__(self, model,device=torch.device("cuda"),dtype=torch.float32):
        super().__init__()
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device
        self.dtype = dtype

        checkpoint_path = "yuvalkirstain/PickScore_v1"
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1"
        self.model = AutoModel.from_pretrained(checkpoint_path).eval().to(self.device, dtype=self.dtype)

        self.target_size =  224
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        

    def forward(self, images, prompts):
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        
        if images.min() < 0: # normalize unnormalized images
            images = ((images / 2) + 0.5).clamp(0, 1)

        inputs = transforms.Resize(self.target_size)(images)
        inputs = self.normalize(inputs).to(self.device,self.dtype)
        image_embeds = self.model.get_image_features(pixel_values=inputs)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        return -1 * scores


def get_optimation_details(args):
    clip_model, clip_preprocess = clip.load("RN50")
    print(clip_preprocess)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    l_func = PickScore(clip_model)
    l_func.eval()
    for param in l_func.parameters():
        param.requires_grad = False
    l_func = torch.nn.DataParallel(l_func).cuda()


    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = None
    operation.other_guidance_func = None

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = l_func
    operation.other_criterion = None

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff
    operation.tv_loss = args.optim_tv_loss

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance
    operation.original_guidance = args.optim_original_conditioning
    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 500
    operation.folder = args.optim_folder

    return operation, l_func

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
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
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
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
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument("--text", default=None)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--trials", default=50, type=int)
    parser.add_argument('--style_folder', default='./data/style_folder/')
    parser.add_argument("--indexes", nargs="+", default=[0, 1, 2], type=int)
    parser.add_argument("--prompt_indexes", nargs="+", default=[0, 1, 2], type=int)

    opt = parser.parse_args()

    results_folder = opt.optim_folder
    create_folder(results_folder)

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()


    sampler = DDIMSamplerWithGrad(model)

    operation, l_func = get_optimation_details(opt)

    # text = []
    # if opt.text != None:
    #     text.append(opt.text)
    # else:
    #     if opt.text_type == 1:
    #         text.append("A colorful photo of a eiffel tower")
    #     elif opt.text_type == 2:
    #         text.append("A fantasy photo of a lonely road")
    #     elif opt.text_type == 3:
    #         text.append("portrait of a woman")
    #     elif opt.text_type == 4:
    #         text.append("A fantasy photo of volcanoes")



    torch.set_grad_enabled(False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    batch_size = opt.batch_size
    ds = ImageFolder(root=opt.style_folder, transform=transform)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                         drop_last=True)

    with open(ASSETS_PATH.joinpath('hps_v2_all_eval.txt'), 'r') as fp:
        prompts = [line.strip() for line in fp.readlines()]

    # prompts = [opt.text]

    for idx, prompt in enumerate(prompts):
        
        if idx not in opt.prompt_indexes:
            continue

        text = [prompt]
        print(text)

        torch.cuda.empty_cache()

        for n, d in enumerate(dl, 0):
            if n in opt.indexes:

                num_images_per_prompt = opt.trials

                offset = 0
                savepath = Path(results_folder).joinpath("images").joinpath(text[0])
                if Path.exists(savepath):

                    images = [x for x in savepath.iterdir() if x.suffix == '.png']
                    num_gen_images = len(images)
                    if num_gen_images >= num_images_per_prompt:
                        print(f'Images found. Skipping prompt.')
                        continue

                    elif num_gen_images < num_images_per_prompt:
                        offset = num_gen_images
                        num_images_per_prompt -= num_gen_images
                        print(f'Found {num_gen_images} images. Generating {num_images_per_prompt} more.')

                if not Path.exists(savepath):
                    Path.mkdir(savepath, exist_ok=True, parents=True)

                uc = None
                if opt.scale != 1.0:
                    uc = model.module.get_learned_conditioning(batch_size * [""])
                c = model.module.get_learned_conditioning(text)

                for multiple_tries in range(num_images_per_prompt):
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, start_zt = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    operated_image=prompt,
                                                    operation=operation)

                    x_samples_ddim = model.module.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    utils.save_image(x_samples_ddim, savepath.joinpath(f'{multiple_tries + offset}.png'))

                    # print(x_samples_ddim.shape)
                    # print(clip_encoded.shape)

                    reward = - operation.loss_func(x_samples_ddim, prompt).squeeze(0)

                    if Path.exists(savepath.joinpath("rewards.json")): # append rewards to file

                        rewards = None
                        with open(savepath.joinpath("rewards.json"), 'r') as fp:
                            rewards = json.load(fp)
                        
                        rewards.extend([reward.detach().cpu().item()])

                        with open(savepath.joinpath("rewards.json"), 'w') as fp:
                            json.dump(rewards, fp)

                    else: # create new rewards file

                        rewards = []
                        rewards.extend([reward.detach().cpu().item()])

                        with open(savepath.joinpath("rewards.json"), 'w') as fp:
                            json.dump(rewards, fp)

if __name__ == "__main__":
    main()
