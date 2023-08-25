#%%

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
from transformer_lens.loading_from_pretrained import MODEL_ALIASES as MA
import argparse
import subprocess

# %%

# "EleutherAI/gpt-neo-125M",

for layer_idx in range(9, 12):
    for head_idx in range(12):
        subprocess.run(["python", "/root/TransformerLens/transformer_lens/rs/arthurs_notebooks/log_attention.py", "--layer-idx", str(layer_idx), "--head-idx", str(head_idx)])