#%%

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
from transformer_lens.loading_from_pretrained import MODEL_ALIASES as MA
import argparse
import subprocess

# %%

# "EleutherAI/gpt-neo-125M",

for MODEL_NAME in ["gpt2-medium"] + [
    model_name for model_name in MA.keys() if "gpt2" in model_name and ("small" in model_name or "medium" in model_name)
]:
    subprocess.run(["python", "/root/TransformerLens/transformer_lens/rs/arthurs_notebooks/sweep_direct_effects.py", "--model-name", MODEL_NAME])