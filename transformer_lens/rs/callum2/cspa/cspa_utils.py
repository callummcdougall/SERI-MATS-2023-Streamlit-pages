# %%
# This contains misc files which don't fit anywhere else.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
for root_dir in [
    os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0] + "SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("/app/seri-mats-2023-streamlit-pages")[0] + "/app/seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
]:
    if os.path.exists(root_dir):
        break
else:
    raise Exception("Couldn't find root dir")
os.chdir(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

from typing import Tuple, List, Union, Dict, Optional
from jaxtyping import Float, Int
import torch as t
from torch import Tensor
from transformer_lens import utils
import numpy as np
import pandas as pd
import einops
from pathlib import Path
from tqdm import tqdm
from transformer_lens import HookedTransformer
from IPython.display import clear_output

from transformer_lens.cautils.utils import get_webtext

ST_HTML_PATH = Path(root_dir + "/media")

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



def devices_are_equal(device_1: Union[str, t.device], device_2: Union[str, t.device]):
    '''
    Helper function, because devices "cuda:0" and "cuda" are actually the same.
    '''
    device_set = set([str(device_1), str(device_2)])
    
    return (len(device_set) == 1) or (device_set == {"cuda", "cuda:0"})



def first_occurrence(array_1D):
    series = pd.Series(array_1D)
    duplicates = series.duplicated(keep='first')
    inverted = ~duplicates
    return inverted.values

def first_occurrence_2d(tensor_2D):
    device = tensor_2D.device
    array_2D = utils.to_numpy(tensor_2D)
    return t.from_numpy(np.array([first_occurrence(row) for row in array_2D])).to(device)




def concat_dicts(d1: Dict[str, Tensor], d2: Dict[str, Tensor]) -> Dict[str, Tensor]:
    '''
    Given 2 dicts, return the dict of concatenated tensors along the zeroth dimension.

    Special case: if d1 is empty, we just return d2.

    Also, we make sure that d2 tensors are moved to cpu.
    '''
    if len(d1) == 0: return d2
    assert d1.keys() == d2.keys()
    return {k: t.cat([d1[k], d2[k]], dim=0) for k in d1.keys()}


def kl_div(
    logits1: Float[Tensor, "... d_vocab"],
    logits2: Float[Tensor, "... d_vocab"],
):
    '''
    Estimates KL divergence D_KL( logits1 || logits2 ), i.e. where logits1 is the "ground truth".

    Each tensor is assumed to have all dimensions be the batch dimension, except for the last one
    (which is a distribution over the vocabulary).

    In our use-cases, logits1 will be the non-ablated version of the model.
    '''

    logprobs1 = logits1.log_softmax(dim=-1)
    logprobs2 = logits2.log_softmax(dim=-1)
    logprob_diff = logprobs1 - logprobs2
    probs1 = logits1.softmax(dim=-1)

    return einops.reduce(
        probs1 * logprob_diff,
        "... d_vocab -> ...",
        reduction = "sum",
    )


def get_result_mean(
    head_list: List[Tuple[int, int]],
    toks: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    minibatch_size: int = 10,
    keep_seq_dim: bool = True,
    verbose: bool = False
) -> Dict[Tuple[int, int], Float[Tensor, "*seq d_model"]]:

    batch_size, seq_len = toks.shape
    layers = list(set([head[0] for head in head_list]))
    result_mean = {head: t.empty((0, seq_len, model.cfg.d_model)) for head in head_list}

    assert batch_size % minibatch_size == 0

    iterator = range(0, batch_size, minibatch_size)
    iterator = tqdm(iterator) if verbose else iterator
    for i in iterator:
        _, cache = model.run_with_cache(
            toks[i: i+minibatch_size],
            return_type=None,
            names_filter=lambda name: name in [utils.get_act_name("result", layer) for layer in layers]
        )
        for layer, head in head_list:
            result = cache["result", layer][:, :, head].mean(0, keepdim=True).cpu() # [1 seq d_model]
            result_mean[(layer, head)] = t.cat([result_mean[(layer, head)], result], dim=0)

    # Remove batch dim, also seq dim if keep_seq_dim is False
    result_mean = {k: v.mean(0) for k, v in result_mean.items()}
    if not keep_seq_dim: result_mean = {k: v.mean(0) for k, v in result_mean.items()}
    
    return result_mean




def process_webtext(
    batch_size: int,
    seq_len: int,
    seed: int = 6,
    indices: Optional[List[int]] = None,
    model: HookedTransformer = None,
    verbose: bool = False,
):
    DATA_STR_ALL = get_webtext(seed=seed)
    DATA_STR_ALL = [parse_str(s) for s in DATA_STR_ALL]
    DATA_STR = []

    count = 0
    for i in range(len(DATA_STR_ALL)):
        num_toks = len(model.to_tokens(DATA_STR_ALL[i]).squeeze())
        if num_toks > seq_len:
            DATA_STR.append(DATA_STR_ALL[i])
            count += 1
        if count == batch_size:
            break
    else:
        raise Exception("Couldn't find enough sequences of sufficient length.")

    DATA_TOKS = model.to_tokens(DATA_STR)
    DATA_STR_TOKS = model.to_str_tokens(DATA_STR)

    if seq_len < 1024:
        DATA_TOKS = DATA_TOKS[:, :seq_len]
        DATA_STR_TOKS = [str_toks[:seq_len] for str_toks in DATA_STR_TOKS]

    DATA_STR_TOKS_PARSED = list(map(parse_str_toks_for_printing, DATA_STR_TOKS))

    clear_output()
    if verbose:
        print(f"Shape = {DATA_TOKS.shape}\n")
        print("First prompt:\n" + "".join(DATA_STR_TOKS[0]))

    return DATA_TOKS, DATA_STR_TOKS_PARSED


