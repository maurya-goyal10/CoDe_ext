from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None, idx=None):
    prompts = _load_lines(path)[low:high]
    if idx == None:
        return random.choice(prompts), {}
    else:
        return prompts[idx%len(prompts)], {}

def hps_v2_all(idx=None):
    return from_file("hps_v2_all.txt")

def simple_animals(idx=None):
    return from_file("simple_animals.txt")

def eval_simple_animals(idx=None):
    return from_file("eval_simple_animals.txt", idx=idx)

def eval_hps_v2_all(idx=None):
    return from_file("hps_v2_all_eval.txt")
