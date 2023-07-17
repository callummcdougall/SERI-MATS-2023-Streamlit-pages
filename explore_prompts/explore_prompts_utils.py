# %%
# This contains misc files which don't fit anywhere else.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
try:
    root_dir = os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts"
    os.chdir(root_dir)
except:
    root_dir = "/app/seri-mats-2023-streamlit-pages/explore_prompts"
    os.chdir(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

from typing import Tuple
from jaxtyping import Float
import torch as t
from torch import Tensor
from transformer_lens import utils
import numpy as np
from pathlib import Path

Head = Tuple[int, int]

NEGATIVE_HEADS = [(10, 7), (11, 10)]

ST_HTML_PATH = Path.cwd() / "media"

def parse_str(s: str):
    doubles = "“”"
    singles = "‘’"
    for char in doubles: s = s.replace(char, '"')
    for char in singles: s = s.replace(char, "'")
    return s

def parse_str_tok_for_printing(s: str):
    s = s.replace("\n", "\\n")
    return s


# %%

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()

# %%



        
