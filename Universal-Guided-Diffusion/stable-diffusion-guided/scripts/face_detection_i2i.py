import argparse, os, sys, glob
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

from torchvision import transforms as T
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
from helper import OptimizerDetails, get_face_text
import clip
import os
import inspect
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.datasets import ImageFolder

ASSETS_PATH = Path("../../BoN/assets/")
NUM_RETRY = 3

# load VAE from diffusers
from diffusers import AutoencoderKL

currhost = os.uname()[1]
if 'housky' in currhost: # shell cluster
    vae = AutoencoderKL.from_pretrained("/glb/data/ptxd_dash/nlasqh/data/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/vae/", use_safetensors=True)
else: # tud/desktop
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", use_auth_token=True)

vae.requires_grad_(False)
vae.eval()

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


class FaceRecognition(nn.Module):
    def __init__(self, fr_crop=False, mtcnn_face=False):
        super().__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        print(self.resnet)
        self.mtcnn = MTCNN(device='cuda')
        self.crop = fr_crop
        self.output_size = 160
        self.mtcnn_face = mtcnn_face

    def extract_face(self, imgs, batch_boxes, mtcnn_face=False):
        image_size = imgs.shape[-1]
        faces = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if not mtcnn_face:
                box = [48, 48, 208, 208]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            elif batch_boxes[i] is not None:
                box = batch_boxes[i][0]
                margin = [
                    self.mtcnn.margin * (box[2] - box[0]) / (self.output_size - self.mtcnn.margin),
                    self.mtcnn.margin * (box[3] - box[1]) / (self.output_size - self.mtcnn.margin),
                ]

                box = [
                    int(max(box[0] - margin[0] / 2, 0)),
                    int(max(box[1] - margin[1] / 2, 0)),
                    int(min(box[2] + margin[0] / 2, image_size)),
                    int(min(box[3] + margin[1] / 2, image_size)),
                ]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            else:
                # crop_face = img[None, :, :, :]
                return None

            faces.append(F.interpolate(crop_face, size=self.output_size, mode='bicubic'))
        new_faces = torch.cat(faces)

        return (new_faces - 127.5) / 128.0

    def get_faces(self, x, mtcnn_face=False):
        img = (x + 1.0) * 0.5 * 255.0
        img = img.permute(0, 2, 3, 1)
        with torch.no_grad():
            batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
            # Select faces
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.mtcnn.selection_method
            )

        img = img.permute(0, 3, 1, 2)
        faces = self.extract_face(img, batch_boxes, mtcnn_face)
        return faces

    def forward(self, x, return_faces=False, mtcnn_face=None):
        x = TF.resize(x, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)

        if mtcnn_face is None:
            mtcnn_face = self.mtcnn_face

        if not self.crop:
            out = self.resnet(x)
        else:
            faces = self.get_faces(x, mtcnn_face=mtcnn_face)
            if faces is None:
                return faces
            out = self.resnet(faces)

        if return_faces:
            if not self.crop:
                faces = self.get_faces(x, mtcnn_face=mtcnn_face)

            return out, faces
        else:
            return out

    def cuda(self):
        self.resnet = self.resnet.cuda()
        self.mtcnn = self.mtcnn.cuda()
        return self

def cycle_cat(dl):
    while True:
        for data in dl:
            yield data[0]

def l1_loss(input, target):
    l = torch.abs(input - target).mean(dim=[1])
    return l

def get_optimation_details(args):
    mtcnn_face = not args.center_face
    print('mtcnn_face')
    print(mtcnn_face)

    guidance_func = FaceRecognition(fr_crop=args.fr_crop, mtcnn_face=mtcnn_face).cuda()
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = guidance_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = l1_loss

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation

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
    parser.add_argument('--optim_mask_type', type=int, default=1)
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
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--text", default=None)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--face_folder', default='./data/face_data')

    parser.add_argument('--fr_crop', action='store_true')
    parser.add_argument('--center_face', action='store_true')
    parser.add_argument("--trials", default=2, type=int)
    parser.add_argument("--indexes", nargs="+", default=[0, 1, 2], type=int)
    parser.add_argument("--prompt_indexes", nargs="+", default=[0, 1, 2], type=int)
    parser.add_argument("--strength", default=1.0, type=float)



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
    operation = get_optimation_details(opt)

    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    batch_size = opt.batch_size

    ds = ImageFolder(root=opt.face_folder, transform=transform)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                                   drop_last=True)


    torch.set_grad_enabled(False)

    # if opt.text != None:
    #     prompt = opt.text
    # else:
    #     prompt = get_face_text(opt.text_type)

    # Load prompts
    with open(ASSETS_PATH.joinpath('face.txt'), 'r') as fp:
        prompts = [line.strip() for line in fp.readlines()]

    # prompts = [opt.text]

    for idx, prompt in enumerate(prompts):
        
        if idx not in opt.prompt_indexes:
            continue

        torch.cuda.empty_cache()

        for n, d in enumerate(dl, 0):
            if n in opt.indexes:

                pending_gen = True # start the loop
                counter = 0
                seed_everything(opt.seed)

                while pending_gen: # loop till it is False
                    
                    ######### changes >>>>>>>>> starts
                    pending_gen = False
                    if counter > 0:
                        seed_everything(opt.seed + counter * 1000)

                    counter += 1
                    ######### changes <<<<<<<<< ends

                    num_images_per_prompt = opt.trials

                    offset = 0
                    savepath = Path(results_folder).joinpath(f'og_img_{n}').joinpath("images").joinpath(prompt)
                    if Path.exists(savepath):

                        images = [x for x in savepath.iterdir() if x.suffix == '.png']
                        num_gen_images = len(images)
                        if num_gen_images == num_images_per_prompt:
                            print(f'Images found. Skipping prompt.')

                        elif num_gen_images < num_images_per_prompt:
                            offset = num_gen_images
                            num_images_per_prompt -= num_gen_images
                            print(f'Found {num_gen_images} images. Generating {num_images_per_prompt} more.')

                    if not Path.exists(savepath):
                        Path.mkdir(savepath, exist_ok=True, parents=True)

                    og_img, _ = d
                    og_img = og_img.cuda()
                    temp = (og_img + 1) * 0.5
                    utils.save_image(temp, f'{results_folder}/og_img_{n}/target.png')

                    with torch.no_grad():
                        og_img_guide, og_img_mask = operation.operation_func(og_img, return_faces=True, mtcnn_face=True)
                        utils.save_image((og_img_mask + 1) * 0.5, f'{results_folder}/og_img_{n}/target_cut.png')

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.module.get_learned_conditioning(batch_size * [""])
                    c = model.module.get_learned_conditioning(batch_size * [prompt])

                    ############## SDEdit #####################
                    timestep = torch.Tensor([int(model.module.num_timesteps * opt.strength)]).to(device).long()
                
                    start_zt = encode(og_img).to(device)

                    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, 
                                        ddim_eta=opt.ddim_eta)
                    noise = torch.randn(start_zt.shape).to(device)
                    if opt.strength > 0.999:
                        start_zt = noise #.half()
                    else:
                        start_zt = add_noise(sampler=sampler, original_samples = start_zt, noise = noise, timesteps = timestep) #.half()
                    ############## SDEdit #####################

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
                                                        operated_image=og_img_guide,
                                                        operation=operation,
                                                        strength=opt.strength,
                                                        start_zt=start_zt)
                        
                        # Compute reward >>>>>>>>>> starts
                        x_samples_ddim_tmp = model.module.decode_first_stage_with_grad(samples_ddim)
                        with torch.no_grad():
                            x_samples_ddim_tmp = operation.operation_func(x_samples_ddim_tmp)

                        if x_samples_ddim_tmp is None:

                            if counter < NUM_RETRY + 1:
                                # retry again
                                print(f'Failed to generate faces. Retry!')
                                pending_gen = True
                                continue

                            else:
                            
                                # save images with inf loss
                                print(f'Failed to generate faces. Save images as is!')

                                x_samples_ddim = model.module.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                                utils.save_image(x_samples_ddim, savepath.joinpath(f'{multiple_tries + offset}.png'))

                                reward = torch.tensor([- torch.inf] * samples_ddim.shape[0])

                        else:
                             # everything is normal
                            reward = - operation.loss_func(x_samples_ddim_tmp, og_img_guide)

                            x_samples_ddim = model.module.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            utils.save_image(x_samples_ddim, savepath.joinpath(f'{multiple_tries + offset}.png'))
                        # Compute reward <<<<<<<<<<< ends

                        if Path.exists(savepath.joinpath("rewards.json")): # append rewards to file

                            rewards = None
                            with open(savepath.joinpath("rewards.json"), 'r') as fp:
                                rewards = json.load(fp)
                            
                            rewards.extend(reward.detach().cpu().tolist())

                            with open(savepath.joinpath("rewards.json"), 'w') as fp:
                                json.dump(rewards, fp)

                        else: # create new rewards file

                            rewards = []
                            rewards.extend(reward.detach().cpu().tolist())

                            with open(savepath.joinpath("rewards.json"), 'w') as fp:
                                json.dump(rewards, fp)

def encode(im):

    transform = T.Compose([T.Resize((512, 512))])

    with torch.no_grad():
        im = transform(im).to(vae.device)
        latent = vae.encode(im).latent_dist.sample()
        latent = vae.config.scaling_factor * latent
    return latent

def add_noise(
        sampler,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:

    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    sampler.alphas_cumprod = sampler.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = sampler.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples

if __name__ == "__main__":
    main()
