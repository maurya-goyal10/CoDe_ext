import os
import torch
from transformers import AutoModel, CLIPProcessor
import torchvision

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class PickScoreScorer(torch.nn.Module):
    def __init__(self, device = device, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        checkpoint_path = "yuvalkirstain/PickScore_v1"
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1"
        self.model = AutoModel.from_pretrained(checkpoint_path).eval().to(self.device, dtype=self.dtype)

        self.target_size =  224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])

    def score(self, images, prompts):
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

        inputs = torchvision.transforms.Resize(self.target_size)(images)
        inputs = self.normalize(inputs).to(self.device,self.dtype)
        image_embeds = self.model.get_image_features(pixel_values=inputs)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        return scores
    
    def loss_fn(self, im_pix, prompts):

        scores = self.score(im_pix, prompts)
        loss =  -1 * scores

        return  loss
