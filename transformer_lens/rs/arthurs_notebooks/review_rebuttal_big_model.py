#%%

import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
import transformers
import datasets

#%%

model_name = "mistralai/Mistral-7B-v0.1"
model = transformers.AutoModel.from_pretrained(model_name) # Hopefully works

#%%

print(model) # What do the modules look like? We'll try to fold the LN bias.

#%%

model = model.to(torch.bfloat16) # Load into lower precision...
gc.collect()
torch.cuda.empty_cache()

#%%

model = model.cuda()
gc.collect()
torch.cuda.empty_cache()

#%%

tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

#%%

mybatch = torch.tensor([tokenizer.encode("Hello, my dog is cute")]) 
# Lol copilot generated
mistral_normal_logits = model(mybatch)[0] # We'll use these later. This takes ages

#%%

# Probably use the `streaming` option on this...
hf_dataset_name = "suolyer/pile_pile-cc"

# Dataset

#%%

hf_iter_dataset = iter(datasets.load_dataset(hf_dataset_name, streaming=True))

#%%

