# python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "a cat wearing glasses"
# python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "A fantasy photo of volcanoes"
python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "cat" "dog" "horse" "monkey" "rabbit" --reward_model "Aesthetic"
python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "snail" "hippopotamus" "cheetah" "crocodile" "lobster" "octopus" --reward_model "Aesthetic"
python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 10 --tt 1 --prompt "snail" "hippopotamus" "cheetah" "crocodile" "lobster" "octopus" --reward_model "Aesthetic"
python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "snail" "hippopotamus" "cheetah" "crocodile" "lobster" "octopus" --fixed_code --seed 2024 --reward_model "Aesthetic" --start_ratio 0.6 --end_ratio 0.2
python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 5 --tt 1 --prompt "snail" "cheetah" --reward_model "Aesthetic" --start_ratio 0.6 --end_ratio 0.2 
python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "monkey" "rabbit" --reward_model "Aesthetic"
python style.py --ddim_steps 100 --n_iter 1 --H 512 --W 512 --scale 5.0 --rho 15 --tt 1 --prompt "cat" --reward_model "Aesthetic"