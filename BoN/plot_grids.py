import os
import json
import matplotlib.pyplot as plt
from PIL import Image

def generate_image_grid(methods, method_name_map, index, base_dir="outputs", prompt_list=None, image_size=(512, 512), save_path="grid.png"):
    """
    Generate a grid of images with rewards: prompts on x-axis, methods on y-axis.

    Parameters:
        methods (list): List of method folder names
        method_name_map (dict): Mapping from method folder name to display name
        index (int): Image index to pick per prompt/method
        base_dir (str): Root directory containing the method folders
        prompt_list (list or None): If None, auto-detect from first method
        image_size (tuple): Size to resize images to (width, height)
        save_path (str): Path to save the final grid image
    """
    if prompt_list is None:
        method0_path = os.path.join(base_dir, methods[0], "images")
        prompt_list = sorted([p for p in os.listdir(method0_path) if not p.endswith(".json")])  # Ignore JSON files

    n_rows = len(methods)
    n_cols = len(prompt_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row, method in enumerate(methods):
        display_method = method_name_map.get(method, method)
        for col, prompt in enumerate(prompt_list):
            ax = axes[row][col]
            ax.axis('off')

            img_path = os.path.join(base_dir, method, "images", prompt, f"{index}.png")
            rewards_path = os.path.join(base_dir, method, "images", prompt, "rewards.json")

            # Set top title as prompt name
            if row == 0:
                ax.set_title(prompt, fontsize=10)
                
            # Load image
            if os.path.exists(img_path):
                img = Image.open(img_path).resize(image_size)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=12)

            # Display reward BELOW the image (with extra spacing)
            reward_text = "Reward: N/A"
            if os.path.exists(rewards_path):
                try:
                    with open(rewards_path, "r") as f:
                        rewards = json.load(f)
                        if isinstance(rewards, list) and index < len(rewards):
                            reward = rewards[index]
                            reward_text = f"Reward: {reward:.3f}"
                except Exception:
                    reward_text = "Reward: Error"
                    
            ax.text(0.5, -0.05, reward_text, ha='center', va='center', transform=ax.transAxes, fontsize=8)  # Increased vertical offset


        # Add method name on the left outside of the row
        fig.text(0.03, (n_rows - row - 0.5) / n_rows, display_method, ha='right', va='center', fontsize=10, rotation=90)

    plt.tight_layout(rect=[0.08, 0, 1, 1])  # leave space on left for method labels
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Grid saved to {save_path}")
    plt.show()


# Example usage:
if __name__ == "__main__":

    # methods = [
    #      'uncond2_aesthetic',
    #      'code_grad4_b5_st7_et3_aesthetic_gs0',
    #      'code_grad1_b5_st6_et2_aesthetic_gs3',
    #      'code40_b5_aesthetic',
    #      'code_grad4_b5_st6_et2_aesthetic_gs3',
    #      'code_grad4_b5_st6_et2_aesthetic_gs5',
    #      ]
    # method_name_map = {
    #     "uncond2_aesthetic": "uncond",
    #     "code40_b5_aesthetic": "CoDe (N=40)",
    #     "code_grad4_b5_st7_et3_aesthetic_gs0": "CoDe (N=4)",
    #     "code_grad1_b5_st6_et2_aesthetic_gs3": "only gradient (gs=0.3)",
    #     "code_grad4_b5_st6_et2_aesthetic_gs3": "CoDe (N=4) + grad (gs=0.3)",
    #     "code_grad4_b5_st6_et2_aesthetic_gs5": "CoDe (N=4) + grad (gs=0.5)",
    # }
    methods = [
         'uncond2_aesthetic',
         'code_grad4_b5_st7_et3_aesthetic_gs0',
         'code40_b5_aesthetic',
         'code_grad4_b5_st6_et2_aesthetic_gs3',
         'DAS_alpha1000_aesthetic'
         ]
    method_name_map = {
        "uncond2_aesthetic": "uncond",
        "code40_b5_aesthetic": "CoDe (N=40)",
        "code_grad4_b5_st7_et3_aesthetic_gs0": "CoDe (N=4)",
        "code_grad4_b5_st6_et2_aesthetic_gs3": "CoDe (N=4) + grad (gs=0.3)",
        'DAS_alpha1000_aesthetic': "DAS (alpha=0.01)"
    }
    index = 0  # Example image index
    generate_image_grid(methods, method_name_map, index,save_path=f"grid_{index}_final.png")
