#%%

import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
import transformers

#%%

model_name = "mistralai/Mistral-7B-v0.1"
model = transformers.AutoModel.from_pretrained(model_name) # Hopefully works

#%%

tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

#%%

mybatch = torch.tensor([tokenizer.encode("Hello, my dog is cute")]) # lol copilot generated
mistral_normal_logits = model(mybatch)[0] # We'll use these later

#%%

print(model) # What do the modules look like? We'll try to fold the LN bias.

#%%

