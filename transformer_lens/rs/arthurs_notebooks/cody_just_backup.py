# In[2]:

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import numpy as np
import pandas as pd
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
# import plotly.express as px
from torch.utils.data import DataLoader
from typing import Union, List, Optional, Callable, Tuple, Dict, Literal, Set
from jaxtyping import Float, Int
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.utils import to_numpy
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache, patching

import plotly.express as px
import circuitsvis as cv
import os, sys

# In[3]:

# Arthur has to change a tad on his setup
from transformer_lens.rs.arthurs_notebooks.path_patching import Node, IterNode, path_patch, act_patch

# In[4]:

from transformer_lens.cautils.plotly_utils import * # covers imshow hist etc

# In[5]:

device

# In[6]:


if not os.path.exists("path_patching.py"):
        get_ipython().system('wget https://github.com/callummcdougall/path_patching/archive/refs/heads/main.zip')
        get_ipython().system("unzip main.zip 'path_patching-main/ioi_dataset.py'")
        get_ipython().system("unzip main.zip 'path_patching-main/path_patching.py'")
        sys.path.append("/path_patching-main")
        os.remove("main.zip")
        os.rename("/path_patching-main/ioi_dataset.py", "ioi_dataset.py")
        os.rename("/path_patching-main/path_patching.py", "path_patching.py")
        os.rmdir("/path_patching-main")

from path_patching import Node, IterNode, path_patch, act_patch


# In[7]:


update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
}

def imshow(tensor, return_fig = False, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "color_continuous_midpoint" not in kwargs_pre:
        fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, **kwargs_pre)
    else:
        fig = px.imshow(utils.to_numpy(tensor), **kwargs_pre)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # things like `xaxis_tickmode` should be applied to all subplots. This is super janky lol but I'm under time pressure
    for setting in ["tickangle"]:
      if f"xaxis_{setting}" in kwargs_post:
          i = 2
          while f"xaxis{i}" in fig["layout"]:
            kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
            i += 1
    fig.update_layout(**kwargs_post)
    fig.show(renderer=renderer)
    if return_fig:
      return fig

def hist(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    names = kwargs_pre.pop("names", None)
    if "barmode" not in kwargs_post:
        kwargs_post["barmode"] = "overlay"
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.histogram(x=tensor, **kwargs_pre).update_layout(**kwargs_post)
    if names is not None:
        for i in range(len(fig.data)):
            fig.data[i]["name"] = names[i // 2]
    fig.show(renderer)


# In[8]:


from plotly import graph_objects as go
from plotly.subplots import make_subplots


# In[9]:


model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,

    device = device
)


# # Model + Dataset Setup

# ## lists

# In[10]:


objects = [
  "perfume",
  "scissors",
  "drum",
  "trumpet",
  "phone",
  "football",
  "token",
  "bracelet",
  "badge",
  "novel",
  "pillow",
  "coffee",
  "skirt",
  "balloon",
  "photo",
  "plate",
  "headphones",
  "flask",
  "menu",
  "compass",
  "belt",
  "wallet",
  "pen",
  "mask",
  "ticket",
  "suitcase",
  "sunscreen",
  "letter",
  "torch",
  "cocktail",
  "spoon",
  "comb",
  "shirt",
  "coin",
  "cable",
  "button",
  "recorder",
  "frame",
  "key",
  "card",
  "canvas",
  "packet",
  "bowl",
  "receipt",
  "pan",
  "report",
  "book",
  "cap",
  "charger",
  "rake",
  "fork",
  "map",
  "soap",
  "cash",
  "whistle",
  "rope",
  "violin",
  "scale",
  "diary",
  "ruler",
  "mouse",
  "toy",
  "cd",
  "dress",
  "shampoo",
  "flashlight",
  "newspaper",
  "puzzle",
  "tripod",
  "brush",
  "cane",
  "whisk",
  "tablet",
  "purse",
  "paper",
  "vinyl",
  "camera",
  "guitar",
  "necklace",
  "mirror",
  "cup",
  "cloth",
  "flag",
  "socks",
  "shovel",
  "cooler",
  "hammer",
  "shoes",
  "chalk",
  "wrench",
  "towel",
  "glove",
  "speaker",
  "remote",
  "leash",
  "magazine",
  "notebook",
  "candle",
  "feather",
  "gloves",
  "mascara",
  "charcoal",
  "pills",
  "laptop",
  "pamphlet",
  "knife",
  "kettle",
  "scarf",
  "tie",
  "goggles",
  "fins",
  "lipstick",
  "shorts",
  "joystick",
  "bookmark",
  "microphone",
  "hat",
  "pants",
  "umbrella",
  "harness",
  "roller",
  "blanket",
  "folder",
  "bag",
  "crate",
  "pot",
  "watch",
  "mug",
  "sandwich",
  "yarn",
  "ring",
  "backpack",
  "glasses",
  "pencil",
  "broom",
  "baseball",
  "basket",
  "loaf",
  "coins",
  "bakery",
  "tape",
  "helmet",
  "bible",
  "jacket"
]


# In[11]:


names = [
  " Sebastian",
  " Jack",
  " Jeremiah",
  " Ellie",
  " Sean",
  " William",
  " Caroline",
  " Cooper",
  " Xavier",
  " Ian",
  " Mark",
  " Brian",
  " Carter",
  " Nicholas",
  " Peyton",
  " Luke",
  " Alexis",
  " Ted",
  " Jan",
  " Ty",
  " Jen",
  " Sophie",
  " Kelly",
  " Claire",
  " Leo",
  " Nolan",
  " Kyle",
  " Ashley",
  " Samantha",
  " Avery",
  " Jackson",
  " Hudson",
  " Rebecca",
  " Robert",
  " Joshua",
  " Olivia",
  " Reagan",
  " Lauren",
  " Chris",
  " Chelsea",
  " Deb",
  " Chloe",
  " Madison",
  " Kent",
  " Thomas",
  " Oliver",
  " Dylan",
  " Ann",
  " Audrey",
  " Greg",
  " Henry",
  " Emma",
  " Josh",
  " Mary",
  " Daniel",
  " Carl",
  " Scarlett",
  " Ethan",
  " Levi",
  " Eli",
  " James",
  " Patrick",
  " Isaac",
  " Brooke",
  " Alexa",
  " Eleanor",
  " Anthony",
  " Logan",
  " Damian",
  " Jordan",
  " Tyler",
  " Haley",
  " Isabel",
  " Alan",
  " Lucas",
  " Dave",
  " Susan",
  " Joseph",
  " Brad",
  " Joe",
  " Vincent",
  " Maya",
  " Will",
  " Jessica",
  " Sophia",
  " Angel",
  " Steve",
  " Benjamin",
  " Eric",
  " Cole",
  " Justin",
  " Amy",
  " Nora",
  " Seth",
  " Anna",
  " Stella",
  " Frank",
  " Larry",
  " Alexandra",
  " Ken",
  " Lucy",
  " Katherine",
  " Leah",
  " Adrian",
  " David",
  " Liam",
  " Christian",
  " John",
  " Nathaniel",
  " Andrea",
  " Laura",
  " Kim",
  " Kevin",
  " Colin",
  " Marcus",
  " Emily",
  " Sarah",
  " Steven",
  " Eva",
  " Richard",
  " Faith",
  " Amelia",
  " Harper",
  " Keith",
  " Ross",
  " Megan",
  " Brooklyn",
  " Tom",
  " Grant",
  " Savannah",
  " Riley",
  " Julia",
  " Piper",
  " Wyatt",
  " Jake",
  " Nathan",
  " Nick",
  " Blake",
  " Ryan",
  " Jason",
  " Chase",]


# In[12]:


# names = [

#     " Mary", " John",
#     " Tom", " James",
#     " Dan", " Sid"  ,
#     " Martin", " Amy",
#     " Cody", " Jay",
#     " Jack", " Jill",
#     " Mark", " Martin",
#     " Sarah", " Emily",
#     " Cole", " George",
#     " Kai", " Bryce",
# ]


# In[13]:


places = [
  "swamp",
  "school",
  "volcano",
  "hotel",
  "subway",
  "arcade",
  "library",
  "island",
  "convent",
  "pool",
  "mall",
  "prison",
  "quarry",
  "temple",
  "ruins",
  "factory",
  "zoo",
  "mansion",
  "tavern",
  "planet",
  "forest",
  "airport",
  "pharmacy",
  "church",
  "park",
  "delta",
  "mosque",
  "valley",
  "casino",
  "pyramid",
  "aquarium",
  "castle",
  "ranch",
  "clinic",
  "theater",
  "gym",
  "studio",
  "station",
  "palace",
  "stadium",
  "museum",
  "plateau",
  "home",
  "resort",
  "garage",
  "reef",
  "lounge",
  "chapel",
  "canyon",
  "brewery",
  "market",
  "jungle",
  "office",
  "cottage",
  "street",
  "gallery",
  "landfill",
  "glacier",
  "barracks",
  "bakery",
  "synagogue",
  "jersey",
  "plaza",
  "garden",
  "cafe",
  "cinema",
  "beach",
  "harbor",
  "circus",
  "bridge",
  "monastery",
  "desert",
  "tunnel",
  "motel",
  "fortress"
]


# ## code
# 

# In[14]:


template = "When{name_A} and{name_B} went to the {place},{name_C} gave the {object} to"


# In[15]:


# prompt: generate a list of prompts using the template. ensure that no prompt uses the same two names

import random

# Create a list of all possible pairs of names
names_list = list(itertools.combinations(names, 2))

# Create a list of prompts
prompts = []
counter_prompts = []


name_pairs = []
#counter_name_pairs = []

for name_pair in names_list:
    name_A, name_B = name_pair
    # Generate a random place
    place = random.choice(places)
    # Generate a random object
    objectA = random.choice(objects)
    # Create a prompt
    prompt = template.format(
        name_A=name_A,
        name_B=name_B,
        place=place,
        name_C=name_B,
        object=objectA,
    )
    prompts.append(prompt)
    name_pairs.append([name_A, name_B])


    # generate three other names that are not name_A and name_B
    other_names = []
    while(len(other_names) != 3):
      new_name = random.choice(names)
      if new_name is not name_A and new_name is not name_B and new_name not in other_names:
        other_names.append(new_name)

    counter_prompts.append(template.format(

        name_A=other_names[0],
        name_B=other_names[1],
        place=place,
        name_C=other_names[2],
        object=objectA,

    ))

# Print the prompts


# In[16]:


# # THIS IS THE REVERSED ONE!

# import random

# # Create a list of all possible pairs of names
# names_list = list(itertools.combinations(names, 2))


# # Create a list of prompts
# prompts = []
# counter_prompts = []


# name_pairs = []
# #counter_name_pairs = []

# for name_pair in names_list:
#     name_A, name_B = name_pair
#     # Generate a random place
#     place = random.choice(places)
#     # Generate a random object
#     objectA = random.choice(objects)
#     # Create a prompt and its reverse
#     prompt = template.format(
#         name_A=name_A,
#         name_B=name_B,
#         place=place,
#         name_C=name_B,
#         object=objectA,
#     )
#     prompts.append(prompt)

#     prompts.append(template.format(
#         name_A=name_B,
#         name_B=name_A,
#         place=place,
#         name_C=name_A,
#         object=objectA,
#     ))
#     name_pairs.append([name_A, name_B])
#     name_pairs.append([name_B, name_A])


#     # generate three other names that are not name_A and name_B
#     other_names = []
#     while(len(other_names) != 3):
#       new_name = random.choice(names)
#       if new_name is not name_A and new_name is not name_B:
#         other_names.append(new_name)

#     counter_prompts.append(template.format(

#         name_A=other_names[0],
#         name_B=other_names[1],
#         place=place,
#         name_C=other_names[2],
#         object=objectA,
#     ))
#     counter_prompts.append(template.format(

#         name_A=other_names[0],
#         name_B=other_names[1],
#         place=place,
#         name_C=other_names[2],
#         object=objectA,
#     ))

# # Print the prompts


# In[17]:


get_ipython().system('nvidia-smi')


# In[18]:


# generate random list of numbers
rand_indices = torch.randint(0, len(prompts), size = (500,))

# indices plus the indices + 1
#double_rand_indices = torch.cat((rand_indices, rand_indices + 1))


# In[19]:


rand_indices


# In[20]:


clean_prompts = [prompts[i] for i in rand_indices]
corrupted_prompts = [counter_prompts[i] for i in rand_indices]
name_answers = [name_pairs[i] for i in rand_indices]


# In[21]:


clean_tokens = model.to_tokens(clean_prompts, prepend_bos = True).cuda()
corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos = True).cuda()
answer_tokens = torch.concat([
    model.to_tokens(names, prepend_bos=False).squeeze(dim=1).unsqueeze(dim=0) for names in name_answers
]).cuda()


# In[22]:


clean_tokens.shape


# In[23]:


index = 1
clean_prompts[index], corrupted_prompts[index], name_answers[index]


# In[24]:


model.reset_hooks()
clean_logits, clean_cache = model.run_with_cache(clean_tokens, prepend_bos = False)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens, prepend_bos = False)


# # chillin
# 

# In[25]:


def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


# In[26]:


clean_per_prompt_diff = logits_to_ave_logit_diff(clean_logits, per_prompt = True)

clean_average_logit_diff = logits_to_ave_logit_diff(clean_logits)
corrupted_average_logit_diff = logits_to_ave_logit_diff(corrupted_logits)

print(clean_average_logit_diff)
print(corrupted_average_logit_diff)


# In[27]:


answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
correct_residual_direction, incorrect_residual_direction = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_direction - incorrect_residual_direction


# In[28]:


def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    '''
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)



    # # some extra code for more sanity checking
    # new_logits = scaled_residual_stack @ model.W_U
    # print(new_logits.shape)
    # new_logits = einops.repeat(new_logits, "batch d_vocab -> batch 1 d_vocab")
    # print(new_logits.shape)
    # print(logits_to_ave_logit_diff(new_logits))

    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size


# In[29]:


answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)

correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
print(f"Logit difference directions shape:", logit_diff_directions.shape)


# In[30]:


model.b_U.shape


# In[31]:


diff_from_unembedding_bias = model.b_U[answer_tokens[:, 0]] -  model.b_U[answer_tokens[:, 1]]


# In[32]:


final_residual_stream: Float[Tensor, "batch seq d_model"] = clean_cache["resid_post", -1]
print(f"Final residual stream shape: {final_residual_stream.shape}")
final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]

print(f"Calculated average logit diff: {(residual_stack_to_logit_diff(final_token_residual_stream, clean_cache, logit_diff_directions) + diff_from_unembedding_bias.mean(0)):.10f}") # <-- okay b_U exists... and matters
print(f"Original logit difference:     {clean_average_logit_diff:.10f}")


# # Helper Functions

# ## Logit Diffs + Gather Important Heads

# In[33]:


def calc_all_logit_diffs(cache):
  clean_per_head_residual, labels = cache.stack_head_results(layer = -1, return_labels = True, apply_ln = False) # per_head_residual.shape = heads batch seq_pos d_model
  # also, for the worried, no, we're not missing the application of LN here since it gets applied in the below function call
  per_head_logit_diff: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache)

  per_head_logit_diff = einops.rearrange(
      per_head_logit_diff,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )

  correct_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = correct_residual_direction)

  correct_direction_per_head_logit = einops.rearrange(
      correct_direction_per_head_logit,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )

  incorrect_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = incorrect_residual_direction)

  incorrect_direction_per_head_logit = einops.rearrange(
      incorrect_direction_per_head_logit,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )

  return per_head_logit_diff, correct_direction_per_head_logit, incorrect_direction_per_head_logit

per_head_logit_diff, correct_direction_per_head_logit, incorrect_direction_per_head_logit = calc_all_logit_diffs(clean_cache)


# In[34]:


top_heads = []
k = 5

flattened_tensor = per_head_logit_diff.flatten().cpu()
_, topk_indices = torch.topk(flattened_tensor, k)
top_layer_arr, top_index_arr = np.unravel_index(topk_indices.numpy(), per_head_logit_diff.shape)

for l, i in zip(top_layer_arr, top_index_arr):
  top_heads.append((l,i))

print(top_heads)


# In[35]:


per_head_logit_diff[11]


# In[36]:


neg_heads = []
neg_indices = torch.nonzero(torch.lt(per_head_logit_diff, -0.1))
neg_heads_list = neg_indices.squeeze().tolist()
for i in neg_heads_list:
  neg_heads.append((i[0], i[1]))

print(neg_heads)


# In[37]:


def display_all_logits(cache, title = "Logit Contributions", comparison = False, return_fig = False, logits = None):

  a,b,c = calc_all_logit_diffs(cache)
  if logits is not None:
    ld = logits_to_ave_logit_diff(logits)
  else:
    ld = 0.00

  if not comparison:
    fig = imshow(
        torch.stack([a,b,c]),
        return_fig = True,
        facet_col = 0,
        facet_labels = [f"Logit Diff - {ld:.2f}", "Correct Direction", "Incorrect Direction"],
        title=title,
        labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1500,
        margin={"r": 100, "l": 100}
    )
  else:

    ca, cb, cc = calc_all_logit_diffs(clean_cache)
    fig = imshow(
        torch.stack([a, b, c, a - ca, b - cb, c - cc]),
        return_fig = True,
        facet_col = 0,
        facet_labels = [f"Logit Diff - {ld:.2f}", "Correct Direction", "Incorrect Direction", "Logit Diff Diff", "Correction Direction Diff", "Incorrect Direction Diff"],
        title=title,
        labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1500,
        margin={"r": 100, "l": 100}
    )


  if return_fig:
    return fig



fig = display_all_logits(clean_cache, title = "Logit Contributions on Clean Dataset", return_fig = True, logits = clean_logits)


# In[38]:


def stare_at_attention_and_head_pat(cache, layer_to_stare_at, head_to_isolate, display_corrupted_text = False, verbose = True, specific = False, specific_index = 0):
  """
  given a cache from a run, displays the attention patterns of a layer, as well as printing out how much the model
  attends to the S1, S2, and IO token
  """

  tokenized_str_tokens = model.to_str_tokens(corrupted_tokens[0]) if display_corrupted_text else model.to_str_tokens(clean_tokens[0])
  attention_patten = cache["pattern", layer_to_stare_at]
  print(f"Layer {layer_to_stare_at} Head {head_to_isolate} Activation Patterns:")


  if not specific:
    S1 = attention_patten.mean(0)[head_to_isolate][-1][2].item()
    IO = attention_patten.mean(0)[head_to_isolate][-1][4].item()
    S2 = attention_patten.mean(0)[head_to_isolate][-1][10].item()
  else:
    S1 = attention_patten[specific_index, head_to_isolate][-1][2].item()
    IO = attention_patten[specific_index, head_to_isolate][-1][4].item()
    S2 = attention_patten[specific_index, head_to_isolate][-1][10].item()


  print("Attention on S1: " + str(S1))
  print("Attention on IO: " + str(IO))
  print("Attention on S2: " + str(S2))
  print("S1 + IO - S2 = " + str(S1 + IO - S2))
  print("S1 + S2 - IO = " + str(S1 + S2 - IO))
  print("S1 - IO - S2 = " + str(S1 - S2 - IO))


  if verbose:
    display(cv.attention.attention_heads(
      tokens=tokenized_str_tokens,
      attention= attention_patten.mean(0) if not specific else attention_patten[specific_index],
      #attention_head_names=[f"L{layer_to_stare_at}H{i}" for i in range(model.cfg.n_heads)],
    ))
  else:
    print(attention_patten.mean(0).shape)

    display(cv.attention.attention_patterns(
      tokens=tokenized_str_tokens,
      attention=attention_patten.mean(0)if not specific else attention_patten[specific_index],
      attention_head_names=[f"L{layer_to_stare_at} H{i}" for i in range(model.cfg.n_heads)],
    ))


# In[ ]:





# In[39]:


def display_corrupted_clean_logits(cache, title = "Logit Contributions", comparison = False, return_fig = False, logits = None):

  a,b,c = calc_all_logit_diffs(cache)
  if logits is not None:
    ld = logits_to_ave_logit_diff(logits)
  else:
    ld = 0.00

  if not comparison:
    fig = imshow(
        torch.stack([a]),
        return_fig = True,
        facet_col = 0,
        facet_labels = [f"Logit Diff - {ld:.2f}"],
        title=title,
        labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1500,
        margin={"r": 100, "l": 100}
    )
  else:

    ca, cb, cc = calc_all_logit_diffs(clean_cache)
    fig = imshow(
        torch.stack([a, a - ca]),
        return_fig = True,
        facet_col = 0,
        facet_labels = [f"New Logit Diff - {ld:.2f}", "New Logit Diff - Clean Logit Diff",],
        title=title,
        labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1000,
        margin={"r": 100, "l": 100}
    )


  if return_fig:
    return fig
  else:
    return a - ca


# In[40]:


heads =  [(9,9), (9,6), (10,0)]
model.reset_hooks() # callum library buggy
def return_item(item):
  return item

model.reset_hooks()
patched_logits = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    patching_metric = return_item,
    verbose = False,
    apply_metric_to_cache = False
)

model.reset_hooks()
noise_sample_ablating_results = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    patching_metric = partial(display_corrupted_clean_logits, title = f"Logits When Sample Ablating in NMHs", comparison = True, logits = patched_logits),
    verbose = False,
    apply_metric_to_cache = True
)


# In[41]:


heads =  [(9,i) for i in range(12)] 
model.reset_hooks() # callum library buggy
def return_item(item):
  return item

model.reset_hooks()
patched_logits = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    patching_metric = return_item,
    verbose = False,
    apply_metric_to_cache = False
)

model.reset_hooks()
all_layernine_noise_patching_results = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    patching_metric = partial(display_corrupted_clean_logits, title = f"Logits When Sample Ablating in Layer 9", comparison = True, logits = patched_logits),
    verbose = False,
    apply_metric_to_cache = True
)


# In[42]:


neg_m_heads = [(10,7), (11,10)]
name_mover_heads = [(9,9), (9,6), (10,0)]
backup_heads = [(9,0), (9,7), (10,1), (10,2), (10,6), (10,10), (11,2), (11,9)]
key_backup_heads = [(10,2), (10,6), (10,10), (11,2)]
strong_neg_backup_heads = [(11,2), (10,2), (10,0), (11,6)]



head_names = ["Negative", "Name Mover", "Backup"]
head_list = [neg_m_heads, name_mover_heads, backup_heads]


# In[43]:


def noising_ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = clean_average_logit_diff,
    corrupted_logit_diff: float = corrupted_average_logit_diff,
) -> float:
    '''
    Given logits, returns how much the performance has been corrupted due to noising.

    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    #print(logits[-1, -1])
    patched_logit_diff = logits_to_ave_logit_diff(logits)
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff))

print(f"IOI metric (IOI dataset): {noising_ioi_metric(clean_logits):.4f}")
print(f"IOI metric (ABC dataset): {noising_ioi_metric(corrupted_logits):.4f}")


# In[44]:


def denoising_ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = clean_average_logit_diff,
    corrupted_logit_diff: float = corrupted_average_logit_diff,
) -> float:
    '''
    We calibrate this so that the value is 1 when performance got restored (i.e. same as IOI dataset),
    and 0 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff(logits)
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff) + 1)


print(f"IOI metric (IOI dataset): {denoising_ioi_metric(clean_logits):.4f}")
print(f"IOI metric (ABC dataset): {denoising_ioi_metric(corrupted_logits):.4f}")


# ## Query Intervention

# In[45]:


def store_activation(
    activation,
    hook: HookPoint,
    where_to_store
):
    """
    takes a storage container where_to_store, and stores the activation in it at a hook
    """""
    where_to_store[:] = activation


# In[46]:


def kq_rewrite_hook(
    internal_value: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    head,
    unnormalized_resid:  Float[Tensor, "batch seq d_model"],
    vector,
    act_name,
    scale = 1,
    position = -1,
    pre_ln = True
):
  """
  replaces keys or queries with a new result which we get from adding a vector to a position at the residual stream
  head: tuple for head to rewrite keys for
  unnormalized_resid: stored unnormalized residual stream needed to recalculated activations
  """

  ln1 = model.blocks[hook.layer()].ln1
  temp_resid = unnormalized_resid.clone()

  if pre_ln:
    temp_resid[:, position, :] = temp_resid[:, position, :] + scale * vector
    normalized_resid = ln1(temp_resid)
  else:
    temp_resid = ln1(temp_resid)
    temp_resid[:, position, :] = temp_resid[:, position, :] + scale * vector
    normalized_resid = temp_resid


  assert act_name == "q" or act_name == "k"
  if act_name == "q":
    W_Q, b_Q = model.W_Q[head[0], head[1]], model.b_Q[head[0], head[1]]
    internal_value[..., head[1], :] = einops.einsum(normalized_resid, W_Q, "batch seq d_model, d_model d_head -> batch seq d_head") + b_Q

  elif act_name == "k":
    W_K, b_K = model.W_K[head[0], head[1]], model.b_K[head[0], head[1]]
    internal_value[..., head[1], :] = einops.einsum(normalized_resid, W_K, "batch seq d_model, d_model d_head -> batch seq d_head") + b_K


# In[47]:


def patch_head_vector(
    head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_indices: int,
    other_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the other cache.
    '''
    for head_index in head_indices:
      head_vector[:, :, head_index] = other_cache[hook.name][:, :, head_index]
    return head_vector


# In[48]:


def patch_ln_scale(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  print("freezing ln1")
  ln_scale = clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]
  return ln_scale


def patch_ln2_scale(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  print("froze ln2")
  ln_scale = clean_cache["blocks." + str(hook.layer()) + ".ln2.hook_scale"]
  return ln_scale


# In[49]:


def causal_write_into_component(act_comp, head, direction, x, pre_ln = True, result_cache_function = None, result_cache_fun_has_head_input = False, freeze_layernorm = False, ablate_heads = []):
  '''
  writes a vector into the component at a given head
  returns new logit differences of run by default, or pass result_cache_funciton to run on cache

  head - tuple for head to intervene in act_comp for
  direction - vector to add to the act_comp in the head
  x - tensor of amount to scale
  '''
  y = torch.zeros(x.shape)
  for i in range(len(x)):
    scale = x[i]
    model.reset_hooks()
    temp = torch.zeros((batch_size, seq_len, model.cfg.d_model)).cuda()
    model.add_hook(utils.get_act_name("resid_pre", head[0]), partial(store_activation, where_to_store = temp))
    if freeze_layernorm:
      model.add_hook("blocks." + str(head[0]) + ".ln1.hook_scale", patch_ln_scale)
    model.add_hook(utils.get_act_name(act_comp, head[0]), partial(kq_rewrite_hook, head = head, unnormalized_resid = temp, vector = direction, act_name = act_comp, scale = scale, pre_ln = pre_ln))


    if len(ablate_heads) != 0:
      for j in ablate_heads:
        model.add_hook(utils.get_act_name("z", j[0]), partial(patch_head_vector, head_indices = [j[1]], other_cache = corrupted_cache))


    hooked_logits, hooked_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()


    if result_cache_function != None:
      if not result_cache_fun_has_head_input:
        y[i] = result_cache_function(hooked_cache)
      else:
        y[i] = result_cache_function(hooked_cache, head)
    else:
      # just calculate logit diff
      y[i] = logits_to_ave_logit_diff(hooked_logits)

  return y


# In[50]:


def graph_lines(results, heads, x, title = "Effect of adding/subtracting direction", xtitle = "Scaling on direction", ytitle = "Logit Diff"):
  fig = px.line(title = title)
  for i in range(len(results)):
    fig.add_trace(go.Scatter(x = x, y = results[i], name = str(heads[i])))

  fig.update_xaxes(title = xtitle)
  fig.update_yaxes(title = ytitle)
  fig.show()


# In[51]:


def get_head_IO_minus_S_attn(cache, head, scores = True):

  layer, h_index = head

  if scores:
    attention_patten = cache["attn_scores", layer]
  else:
    attention_patten = cache["pattern", layer]
  S1 = attention_patten.mean(0)[h_index][-1][2].item()
  IO = attention_patten.mean(0)[h_index][-1][4].item()
  S2 = attention_patten.mean(0)[h_index][-1][10].item()

  return IO - S1 - S2


def get_head_IO_minus_just_S1_attn(cache, head, scores = True):

    layer, h_index = head

    if scores:
      attention_patten = cache["attn_scores", layer]
    else:
      attention_patten = cache["pattern", layer]
    S1 = attention_patten.mean(0)[h_index][-1][2].item()
    IO = attention_patten.mean(0)[h_index][-1][4].item()
    S2 = attention_patten.mean(0)[h_index][-1][10].item()

    return IO - S1

def get_head_last_token(cache, head):
  layer, h_index = head
  return cache["pattern", layer][:, h_index, -1, :]


def get_head_attn(cache, head, token, scores = True, mean = True):

  layer, h_index = head

  if scores:
    attention_patten = cache["attn_scores", layer]
  else:
    attention_patten = cache["pattern", layer]


  if mean:
    if token == "S1":
      return attention_patten.mean(0)[h_index][-1][2].item()
    elif token == "IO":
      return attention_patten.mean(0)[h_index][-1][4].item()
    elif token == "S2":
      return attention_patten.mean(0)[h_index][-1][10].item()
    elif token == "BOS":
      return attention_patten.mean(0)[h_index][-1][0].item()
    else:
      print("RAHHHHH YOU MISSTYPED SOMETHING")

  else:
    if token == "S1":
      return attention_patten[:, h_index, -1, 2]
    elif token == "IO":
      return attention_patten[:, h_index, -1, 4]
    elif token == "S2":
      return attention_patten[:, h_index, -1, 10]
    elif token == "BOS":
      return attention_patten[:, h_index, -1, 0]
    else:
      print("RAHHHHH YOU MISSTYPED SOMETHING")

      
def patch_head_vector(
    head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_indices: int,
    other_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the other cache.
    '''
    for head_index in head_indices:
      head_vector[:, :, head_index] = other_cache[hook.name][:, :, head_index]
    return head_vector

def get_attn_results_into_head_dirs(heads, direction, scale_amounts, ablate_heads = [], freeze_ln = False, only_S1 = False):
  io_attn_postln_nmh_results = []
  for i in range(len(heads)):
    io_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln, result_cache_function = partial(get_head_attn, token = "IO"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))


  s1_attn_postln_nmh_results = []
  for i in range(len(heads)):
    s1_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = partial(get_head_attn, token = "S1"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))

  s2_attn_postln_nmh_results = []
  for i in range(len(heads)):
    s2_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = partial(get_head_attn, token = "S2"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))

  diff_results = []
  if not only_S1:
    for i in range(len(heads)):
      diff_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                          pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = get_head_IO_minus_S_attn, result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))
  else:
    for i in range(len(heads)):
      diff_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                          pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = get_head_IO_minus_just_S1_attn, result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))


  bos_attn_postln_nmh_results = []
  for i in range(len(heads)):
    bos_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = partial(get_head_attn, token = "BOS"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))

  return [io_attn_postln_nmh_results, s1_attn_postln_nmh_results, s2_attn_postln_nmh_results, diff_results, bos_attn_postln_nmh_results]


# In[52]:


IO_unembed_direction = model.W_U.T[clean_tokens][:, 4, :]


# # Unembedding to Not Ratios

# In[67]:


model.set_use_attn_result(True)


# In[101]:


def get_projection(from_vector, to_vector):
    dot_product = einops.einsum(from_vector, to_vector, "batch d_model, batch d_model -> batch")
    #print("Average Dot Product of Output Across Batch: " + str(dot_product.mean(0)))
    length_of_from_vector = einops.einsum(from_vector, from_vector, "batch d_model, batch d_model -> batch")
    length_of_vector = einops.einsum(to_vector, to_vector, "batch d_model, batch d_model -> batch")
    
    


    projected_lengths = (dot_product) / (length_of_vector)
    #print( einops.repeat(projected_lengths, "batch -> batch d_model", d_model = model.cfg.d_model)[0])
    projections = to_vector * einops.repeat(projected_lengths, "batch -> batch d_model", d_model = to_vector.shape[-1])
    return projections


# In[118]:


a = torch.Tensor([[-1, 1]])
b = torch.Tensor([[1, 1]])
print(get_projection(a, b))


# In[69]:


import torch.nn.functional as F

def compute_cosine_similarity(tensor1, tensor2):
    # Compute cosine similarity
    similarity = F.cosine_similarity(tensor1, tensor2, dim=1)
    return similarity


# In[119]:


def project_vector_operation(
    original_resid_stream: Float[Tensor, "batch seq head_idx d_model"],
    hook: HookPoint,
    vector: Float[Tensor, "batch d_model"],
    position = -1,
    heads = [], # array of ints
    scale_proj = 1,
    project_only = False
) -> Float[Tensor, "batch n_head pos pos"]:
  '''
  Function which gets orthogonal projection of residual stream to a vector, and either subtracts it or keeps only it
  '''
  #print("RUH")
  for head in heads:
    projections = get_projection(original_resid_stream[:, position, head, :], vector)
    if project_only:
      original_resid_stream[:, position, head, :] = projections * scale_proj
    else:
      original_resid_stream[:, position, head, :] = (original_resid_stream[:, position, head, :] - projections) * scale_proj #torch.zeros(original_resid_stream[:, position, head, :].shape)#
  
  return original_resid_stream


# In[120]:


# get ldds when intervening and replacing with directions of corrupted runs
def project_away_component_and_replace_with_something_else(
    original_resid_out: Float[Tensor, "batch seq head_idx d_model"],
    hook: HookPoint,
    project_away_vector: Float[Tensor, "batch d_model"],
    replace_vector : Float[Tensor, "batch d_model"],
    position = -1,
    heads = [], # array of ints,
    project_only = False # whether to, instead of projecting away the vector, keep it!
) -> Float[Tensor, "batch n_head pos pos"]:
    '''
    Function which gets removes a specific component (or keeps only it, if project_only = True) of the an output of a head and replaces it with another vector
    '''
    # right now this projects away the IO direction!
    assert project_away_vector.shape == replace_vector.shape and len(project_away_vector.shape) == 2

    for head in heads:
        
        head_output = original_resid_out[:, position, head, :]
        projections = get_projection(head_output, project_away_vector)

        if project_only:
            resid_without_projection =  projections
        else:
            resid_without_projection = (head_output - projections) 

        updated_resid = resid_without_projection + replace_vector
        original_resid_out[:, position, head, :] = updated_resid

    return original_resid_out


# In[127]:


unembed_io_directions = model.tokens_to_residual_directions(answer_tokens[:, 0])
unembed_s_directions = model.tokens_to_residual_directions(answer_tokens[:, 1])
unembed_diff_directions = unembed_io_directions - unembed_s_directions
ca, cb, cc = calc_all_logit_diffs(clean_cache)


# In[128]:


def patch_last_ln(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  print("froze lnfinal")
  ln_scale = clean_cache["ln_final.hook_scale"]
  return ln_scale


# In[129]:


def project_stuff_on_heads(project_heads, project_only = False, scale_proj = 1, output = "display_logits", freeze_ln = False):
    model.reset_hooks()

    # project_heads is a list of tuples (layer, head). for each layer, write a hook which projects all the heads from the layer
    for layer in range(model.cfg.n_layers):
        key_heads = [head[1] for head in project_heads if head[0] == layer]
        if len(key_heads) > 0:
            #print(key_heads)
            model.add_hook(utils.get_act_name("result", layer), partial(project_vector_operation, vector = unembed_diff_directions, heads = key_heads, scale_proj = scale_proj, project_only = project_only))
            
    if freeze_ln:
                # freeze ln
        for layer in [9,10,11]:
            model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
            model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)
        model.add_hook("ln_final.hook_scale", patch_last_ln)

    hooked_logits, hooked_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    if output == "display_logits":
        return display_all_logits(hooked_cache, comparison=True, logits = hooked_logits, title = f"Projecting {('only' if project_only else 'away')} IO direction in heads {project_heads}")
    elif output == "get_ldd":
        a,_,_ = calc_all_logit_diffs(hooked_cache)
        
        return a - ca


# In[166]:


def compare_intervention_ldds_with_sample_ablated(all_ldds, ldds_names, heads = key_backup_heads, just_logits = False):
    results = torch.zeros((len(all_ldds), len(heads)))
    

    if just_logits:
        for ldd_index, compare_ldds in enumerate(all_ldds):
            for i, head in enumerate(heads):
                #print(head)
                results[ldd_index, i] = ((compare_ldds[head[0], head[1]]).item()) # / noise_sample_ablating_results[head[0], head[1]]).item())
    else:
        for ldd_index, compare_ldds in enumerate(all_ldds):
            for i, head in enumerate(heads):
                #print(head)
                results[ldd_index, i] = ((compare_ldds[head[0], head[1]] / noise_sample_ablating_results[head[0], head[1]]).item())
    
    imshow(
        results,
        #facet_col = 0,
        #labels = [f"Head {head}" for head in key_backup_heads],
        title=f"The {'Ratio of Backup (Logit Diff Diff)' if not just_logits else 'Logit Diff Diffs'} of Intervention" + ("to Sample Ablation Backup" if not just_logits else ""),
        labels={"x": "Receiver Head", "y": "Intervention", "color": "Ratio of Logit Diff Diff to Sample Ablation" if not just_logits else "Logit Diff Diff"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        # range of y-axis color from 0 to 2
        #color_continuous_scale="mint",
        color_continuous_midpoint=1 if not just_logits else 0,
        # give x-axis labels
        x = [str(head) for head in heads],
        y = ldds_names,
        border=True,
        width=1100,
        height = 600,
        margin={"r": 100, "l": 100},
        # show the values of the results above the heatmap
        text_auto = True,

    )


# In[167]:


project_stuff_on_heads([(9,6), (9,9), (10,0)], project_only = False, scale_proj = 1, output = "display_logits", freeze_ln=True)


# In[168]:


# get projection stuff

zero_ablate_all_heads_ldds = project_stuff_on_heads([(9,6), (9,9), (10,0)], project_only = True, scale_proj = 0, output = "get_ldd", freeze_ln=True)
project_only_io_direction = project_stuff_on_heads([(9,6), (9,9), (10,0)], project_only = True, scale_proj = 1, output = "get_ldd", freeze_ln=True)
project_away_io_direction = project_stuff_on_heads([(9,6), (9,9), (10,0)], project_only = False, scale_proj = 1, output = "get_ldd", freeze_ln=True)

# rerun but in terms of project and replace function
# model.reset_hooks()
# separate_zero_ablate_all_heads_ldds = 


# get results from replacing all IO directions

# In[169]:


target_heads = [(9,6), (9,9), (10,0)]
model.reset_hooks()
for layer in [9,10,11]:
            model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
            model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)
model.add_hook("ln_final.hook_scale", patch_last_ln)

for head in target_heads:
    
    # get the output of head on CORRUPTED RUN
    W_O_temp = model.W_O[head[0], head[1]]
    layer_z = corrupted_cache[utils.get_act_name("z", head[0])]
    layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
    output_head = layer_result[:, -1, head[1], :]

    # get projection of CORRUPTED HEAD OUTPUT onto IO token
    corrupted_head_only_IO_output = get_projection(output_head, unembed_diff_directions)
    
    # add hook to now replace with this corrupted IO direction
    model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = unembed_diff_directions, heads = [head[1]], replace_vector = corrupted_head_only_IO_output))

replace_with_new_IO_logits, replace_with_new_IO_cache = model.run_with_cache(clean_tokens)

replace_all_IOs_ldds = calc_all_logit_diffs(replace_with_new_IO_cache)[0] - ca
model.reset_hooks()


# In[170]:


# display logit diffs
display_all_logits(replace_with_new_IO_cache, comparison=True, logits = replace_with_new_IO_logits, title = f"Replacing IO direction in heads {target_heads}")


# get results from replacing all perp to IO directions

# In[171]:


target_heads = [(9,6), (9,9), (10,0)]
model.reset_hooks()
for layer in [9,10,11]:
            model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
            model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)

model.add_hook("ln_final.hook_scale", patch_last_ln)
for head in target_heads:
    
    # get the output of head on CORRUPTED RUN
    W_O_temp = model.W_O[head[0], head[1]]
    layer_z = corrupted_cache[utils.get_act_name("z", head[0])]
    layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
    output_head = layer_result[:, -1, head[1], :]

    
    # get projection of CORRUPTED HEAD OUTPUT onto IO perp token
    corrupted_head_only_IO_output = get_projection(output_head, unembed_diff_directions)
    everything_else_but_that = output_head - corrupted_head_only_IO_output
    
    # add hook to now replace with this corrupted IO perp direction
    model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = unembed_diff_directions, heads = [head[1]], replace_vector = everything_else_but_that, project_only = True))

replace_with_new_perp_IO_logits, replace_with_new_perp_IO_cache = model.run_with_cache(clean_tokens)
replace_all_perp_IOs_ldds = calc_all_logit_diffs(replace_with_new_perp_IO_cache)[0] - ca
model.reset_hooks()


# In[172]:


display_all_logits(replace_with_new_perp_IO_cache, comparison=True, logits = replace_with_new_perp_IO_logits, title = f"Projecting weird lol")


# In[173]:


compare_intervention_ldds_with_sample_ablated([ca - ca, zero_ablate_all_heads_ldds, project_only_io_direction, replace_all_perp_IOs_ldds,  project_away_io_direction, replace_all_IOs_ldds, noise_sample_ablating_results],
                                               ["Clean Run", "Zero Ablation of NMHs", "Project Only LD Direction (Zero ⊥ LD direction)", "Replace ⊥ LD directions with Corrupted ⊥ LD directions", "Project Away LD Direction (Zero LD direction)", "Replace LD directions with Corrupted LD directions",  "Sample Ablation of NMHs"],
                                               heads = key_backup_heads + neg_m_heads, just_logits = True)


# In[ ]:





# In[ ]:


for index, i in enumerate([zero_ablate_all_heads_ldds, project_only_io_direction, replace_all_perp_IOs_ldds,  project_away_io_direction, replace_all_IOs_ldds, noise_sample_ablating_results]):
    imshow(i,
           title = ["Zero Ablation of NMHs", "Project Only IO Direction (Zero ⊥ IO direction)", "Replace ⊥ IO directions with Corrupted ⊥ IO directions", "Project Away IO Direction (Zero IO direction)", "Replace IO directions with Corrupted IO directions",  "Sample Ablation of NMHs"][index])


# In[ ]:


# find cosine similarity of 9.0 output and IO unembedding
for head in neg_m_heads:
    print("Mean Cossim similarity between IO unembedding and " + str(head) + ":")
    W_O_temp = model.W_O[head[0], head[1]]
    layer_z = clean_cache[utils.get_act_name("z", head[0])]
    layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
    output_head = layer_result[:, -1, head[1], :]

    # get projection of CORRUPTED HEAD OUTPUT onto IO token
    corrupted_head_only_IO_output = compute_cosine_similarity(output_head, unembed_diff_directions)
    print(corrupted_head_only_IO_output.mean(0))


# In[ ]:


model.set_use_attn_result(False)

