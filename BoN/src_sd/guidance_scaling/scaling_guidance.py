import torch

def set_scale(grad,correction=None,target_guidance=None,guidance_scale=None,method="FreeDoM"):
    if method == "FreeDoM":
        # grad_norm = (grad * grad).mean(axis=(1,2,3)).sqrt()
        # target_guidance = (correction * correction).mean(axis=(1,2,3)).sqrt() * guidance_scale / (grad_norm + 1e-8) * target_guidance 
        # bool_target_guidance = target_guidance > 150.0
        # target_guidance[bool_target_guidance] = 150.0
        
        grad_norm = (grad * grad).mean().sqrt().item()
        target_guidance = (correction * correction).mean().sqrt().item() * guidance_scale / (grad_norm + 1e-8) * target_guidance 
        if target_guidance > 150.0: target_guidance = 150.0
        return target_guidance