#%%

from transformer_lens.cautils.notebook import *
from transformer_lens.cautils.ioi_dataset import NAMES
from transformer_lens.rs.callum2.utils import get_effective_embedding

#%%

model = HookedTransformer.from_pretrained("gpt2")

#%%

W_EE = get_effective_embedding(model)["W_EE (including MLPs)"]

# %%
