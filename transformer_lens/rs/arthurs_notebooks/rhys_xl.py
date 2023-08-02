#%%

from transformer_lens.cautils.notebook import *

# %%

model = HookedTransformer.from_pretrained("gpt2-xl")

# %%

words = [" Monday", " Tuesday", " Wednesday", " January", "February", " one"]
tokens = torch.tensor([model.to_tokens(word, prepend_bos=False) for word in words])

#%%

model.reset_hooks()

model.add_hook(
    get_act_name("pattern", 0), 
    lock_attn,
)
model.add_hook(
    "hook_pos_embed",
    lambda x, hook: x * 0,
)
resid_post = model.run_with_cache(
    tokens.unsqueeze(0),
    names_filter = lambda name: name == "blocks.0.hook_resid_post",
)[1]["blocks.0.hook_resid_post"]

# %%

layer = 29
head_idx = 15

W_V = model.W_V[layer, head_idx]
W_O = model.W_O[layer, head_idx]

wved = einops.einsum(
    resid_post,
    W_V,
    "batch one d_model, d_model d_head -> batch one d_head",
)

woed = einops.einsum(
    wved,
    W_O, 
    "batch one d_head, d_head d_model -> batch one d_model",
)

logit_lensed = einops.einsum(
    woed,
    model.W_U,
    "batch one d_model, d_model vocab -> batch one vocab",
)

#%%

top_logits = torch.topk(
    logit_lensed, 
    dim=-1,
    k = 10,
).indices

#%%

for i in range(6):
    print(model.to_str_tokens(top_logits[0, i]))
# %%

model = HookedTransformer.from_pretrained("EleutherAI/gpt-J-6B")

# %%

tokens = model.to_tokens("Hello, world!")
assert tokens.shape == (1, 4) # [batch, seq] with prepended BOS

logits = model("Hello, world!")
log_probs = logits.log_softmax(-1)
assert abs(log_probs.norm() - 1_000_000) < 1e-5

#%%