# %%
# This contains misc files which don't fit anywhere else.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
from typing import Tuple, List
from jaxtyping import Float
import torch as t
from torch import Tensor
from transformer_lens import utils
import numpy as np
from pathlib import Path

str_path_before_transformer_lens = os.getcwd().split("transformer_lens")[0]
ST_HTML_PATH = Path(f"{str_path_before_transformer_lens}/transformer_lens/rs/callum2/st_page/media")
if not(ST_HTML_PATH.exists()):
    str_path_before_transformer_lens = os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0]
    ST_HTML_PATH = Path(f"{str_path_before_transformer_lens}/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media")
assert ST_HTML_PATH.exists(), f"ST_HTML_PATH doesn't exist: {ST_HTML_PATH}"

Head = Tuple[int, int]

NEGATIVE_HEADS = [(10, 7), (11, 10)]

def parse_str(s: str):
    doubles = "“”"
    singles = "‘’"
    for char in doubles: s = s.replace(char, '"')
    for char in singles: s = s.replace(char, "'")
    return s

def parse_str_tok_for_printing(s: str):
    s = s.replace("\n", "\\n")
    return s

def parse_str_toks_for_printing(s: List[str]):
    return list(map(parse_str_tok_for_printing, s))


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

def create_title_and_subtitles(
    title: str,
    subtitles: List[str],
) -> str:
    return f"{title}<br><span style='font-size:13px'>{'<br>'.join(subtitles)}</span>"



