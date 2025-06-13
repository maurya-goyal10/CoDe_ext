import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import textwrap

def plot_image_grid(method_name, n, save_path=None,skip=0):
    base_path = os.path.join(method_name, "images")
    prompts = sorted(os.listdir(base_path))
    
    num_prompts = len(prompts)
    fig, axes = plt.subplots(nrows=n*2, ncols=num_prompts, figsize=(3*num_prompts, 3*n), gridspec_kw={'height_ratios': [10, 1]*n})
    method_name_split = method_name.split('_')
    title = f"{method_name_split[0]} {method_name_split[2]} {method_name_split[1]}"
    plt.suptitle(title,fontsize=16, y=1.02)

    for col, prompt in enumerate(prompts):
        prompt_path = os.path.join(base_path, prompt)
        rewards_path = os.path.join(prompt_path, "rewards.json")
        
        with open(rewards_path, "r") as f:
            rewards = json.load(f)

        # Set prompt title above first image in column
        wrapped_prompt = textwrap.fill(prompt,30)
        axes[0][col].set_title(wrapped_prompt, fontsize=12,wrap=True)

        for i in range(n):
            img_idx = i
            img_path = os.path.join(prompt_path, f"{skip+img_idx}.png")

            ax_img = axes[i*2][col]
            ax_text = axes[i*2 + 1][col]

            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax_img.imshow(img)
                ax_img.axis("off")
                ax_text.text(0.5, 0.5, f"Reward: {rewards[img_idx]:.4f}", ha="center", va="center", fontsize=10)
            else:
                ax_img.axis("off")
                ax_text.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10)

            ax_text.axis("off")

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # reserve space for title
    fig.suptitle(title, fontsize=16)
    plt.savefig(save_path, dpi=300)
    print(f"Saved the figure in {save_path}")

if __name__ == "__main__":
    method = "code4_multinomial_temp200000_b5_aesthetic"
    savepath = f"./{method}/grid_of_results"
    savepath_2 = f"./{method}/grid_of_results_2"
    plot_image_grid(method, n=5,save_path=savepath)
    plot_image_grid(method, n=5,save_path=savepath,skip=5)
