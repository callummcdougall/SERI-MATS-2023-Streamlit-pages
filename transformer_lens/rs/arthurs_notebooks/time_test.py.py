#%%

import transformer_lens
import time

#%%

model = transformer_lens.HookedTransformer.from_pretrained("pythia-2.8B")

# %%

t0 = time.time()
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        w_k = model.W_K[layer, head]
        w_k.norm() # some random calculation
print(time.time() - t0)

# %%
