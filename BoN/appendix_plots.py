import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

def plots():

    task = 'compress'

    # _METHODS_FACE = {
    #     'BoN' : f'bon40_{task}',
    #     'DPS (2023)' : f'grad12999999999999998_{task}' if task == 'facedetector' else f'grad12999999999999998_{task}',
    #     'UG (2024)': 'test_face_20000' if task == 'facedetector' else 'test_style_6' if task == 'styletransfer' else 'test_stroke_6',
    #     'SVDD-PM (2024)': f'code40_b1_{task}',
    #     'SDEdit' : f'i2i_r7_{task}' if task == 'facedetector' else f'i2i_r6_{task}', 
    #     'CoDe (Ours)' : f'code100_b5_{task}',
    #     'C-UG (2024)': 'test_face_i2i_20000_r7' if task == 'facedetector' else 'test_style_i2i_6_r6' if task == 'styletransfer' else 'test_stroke_i2i_6_r6',
    #     'C-CoDe (Ours)' : f'c_code100_b5_r7_{task}' if task == 'facedetector' else f'c_code100_b5_r6_{task}',
    # }

    # _METHODS_FACE = {
    #     'C-UG (0.5)': 'test_face_i2i_20000_r5' if task == 'facedetector' else 'test_style_i2i_6_r5',
    #     'C-UG (0.6)': 'test_face_i2i_20000_r6' if task == 'facedetector' else 'test_style_i2i_6_r6',
    #     'C-UG (0.7)': 'test_face_i2i_20000_r7' if task == 'facedetector' else 'test_style_i2i_6_r7',
    #     'C-UG (0.8)': 'test_face_i2i_20000_r8' if task == 'facedetector' else 'test_style_i2i_6_r8',
    # }
    _METHODS_FACE = {
        'C-UG (0.5)': 'test_face_i2i_20000_r5' if task == 'facedetector' else 'test_style_i2i_6_r5',
        'C-UG (0.6)': 'test_face_i2i_20000_r6' if task == 'facedetector' else 'test_style_i2i_6_r6',
        'C-UG (0.7)': 'test_face_i2i_20000_r7' if task == 'facedetector' else 'test_style_i2i_6_r7',
        'C-UG (0.8)': 'test_face_i2i_20000_r8' if task == 'facedetector' else 'test_style_i2i_6_r8',
    }

    base_path = Path('outputs_img_abla')

    firstKey = list(_METHODS_FACE.keys())[0]
    target_dirs = [x for x in base_path.joinpath(_METHODS_FACE[firstKey]).iterdir() if Path.is_dir(x) and x.stem != 'plots']
    for target_dir in target_dirs:

        prompt_dirs = [x for x in target_dir.joinpath('images').iterdir() if Path.is_dir(x)]
        for prompt_dir in prompt_dirs:
            
            save_path = base_path.joinpath('plots').joinpath(task).joinpath(target_dir.stem).joinpath(prompt_dir.stem)
            if not Path.exists(save_path):
                Path.mkdir(save_path, exist_ok=True, parents=True)

            for it in range(4):

                fig, ax = plt.subplots(12, 5, figsize=(5,15))

                # target_path = Path('appendix/sdedit/style_2/images/target.png')
                if task == 'facedetector':
                    target_path = target_dir.joinpath('target.png')
                else:
                    target_path = base_path.joinpath(_METHODS_FACE[firstKey]).joinpath(f'{target_dir.stem}.png')
                target = Image.open(target_path)

                ax[0][0].set_title('target')
                ax[0][0].imshow(target)
                ax[0][0].axis('equal')
                ax[0][0].axis('off')

                for i in range(11):
                    ax[i+1][0].axis('off')

                for j, method in enumerate(_METHODS_FACE.keys()):
                    
                    # img_path = base_path.joinpath(_METHODS_FACE[method]).joinpath('style_2/A colorful photo of a eiffel tower')
                    img_path = base_path.joinpath(_METHODS_FACE[method]).joinpath(target_dir.stem)\
                        .joinpath('images').joinpath(prompt_dir.stem)

                    # print(img_path)
                    if not Path.exists(img_path):
                        print(img_path)
                        for i in range(12):
                            ax[i][j+1].axis('equal')
                            ax[i][j+1].axis('off')
                            if i == 0:
                                ax[i][j+1].set_title(method[:4])

                        continue

                    imgs = [x for x in img_path.iterdir() if Path.is_file(x) and x.suffix == '.png']
                    # print(len(imgs))
                    
                    for i in range(12):
                        if i < len(imgs):
                            ax[i][j+1].imshow(Image.open(imgs[i+(it*5)]))
                        ax[i][j+1].axis('equal')
                        ax[i][j+1].axis('off')

                        if i == 0:
                            ax[i][j+1].set_title(method)

                    # fig.suptitle(f'{target_dir.stem} : {prompt_dir.stem}')
                    # plt.tight_layout()
                    # plt.show()

                # plt.show()
                plt.tight_layout()
                # print(plt_save)
                plt.savefig(save_path.joinpath(f'res{it}_upd.png'), dpi=300)
                plt.close()

        #         break

        #     break

        # break

if __name__ == '__main__':
    plots()