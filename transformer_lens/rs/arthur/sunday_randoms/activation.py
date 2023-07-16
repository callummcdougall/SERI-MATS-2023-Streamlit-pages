# %% [markdown]

from transformer_lens import HookedTransformer
import torch as t
import numpy as np
import sys
from functools import partial

model = HookedTransformer.from_pretrained(
    "gpt2-XL",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)

#%%

steering_tokens = model.to_tokens("Love ")

# %%

generation_tokens = model.to_tokens("I hate you because")

#%%

model.reset_hooks()
normal_output = model.generate(generation_tokens, max_new_tokens=20)
print(model.to_string(normal_output))

# %%

six_pre = "blocks.6.hook_resid_pre"
model.reset_hooks()
cached_value = model.run_with_cache(
    steering_tokens,
    names_filter = lambda name: name == six_pre,
)[1]["blocks.6.hook_resid_pre"]

second_cached_value = model.run_with_cache(
    model.to_tokens("Hate"),
    names_filter = lambda name: name == six_pre,
)[1]["blocks.6.hook_resid_pre"]


# %%

def activation_addition_hook(z, hook, value, coefficient=1.0):
    assert 3 == len(z.shape) == len(value.shape), f"z.shape: {z.shape}, value.shape: {value.shape}"
    z[:, :value.shape[1], :] += coefficient * value
    return z

#%%

modes = [0 for _ in range(10000)] + [1 for _ in range(10000)]
np.random.shuffle(modes)

for mode in modes:
    coeffs = [mode, 1.0 - mode]
    for coeff in coeffs:
        model.reset_hooks()
        model.add_hook(
            six_pre,
            partial(activation_addition_hook, value=cached_value - coeff * second_cached_value, coefficient=5.0),
        )
        new_generation_tokens = model.generate(generation_tokens, max_new_tokens=20, use_past_kv_cache=False)
        print(model.to_string(new_generation_tokens))
    # flush
    sys.stdout.flush()

    input("Hiding which output is which now...")
    print(("First is love only") if mode == 0 else "First is love-hate")

#%%
