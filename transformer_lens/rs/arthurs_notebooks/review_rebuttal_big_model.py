#%%

import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
import transformers
# from einshape.src.pytorch.pytorch_ops import einshape
from transformer_lens.cautils.notebook import *
import torch

# from transformer_lens.cautils.notebook import *
# from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
# import datasets

import gc

# #%% # Note 3 minute delay
# model_name = "mistralai/Mistral-7B-v0.1"

#%%

model_name = "huggyllama/llama-7b"
torch.set_grad_enabled(False)

#%%

from transformer_lens import HookedTransformer
import torch

#%%

# from_pretrained version 4 minutes
if "llama" in model_name.lower() or "mistral" in model_name.lower():
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained_no_processing("llama-7b", hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=hf_tokenizer)

else:
    model = HookedTransformer.from_pretrained_no_processing(model_name, fold_ln=False, center_writing_weights=False, center_unembed=False).cuda()

#%%

print(model)
# What do the modules look like? We'll try to fold the LN bias

#%% # Takes 90 secs. And in TL >150 seconds

model = model.to(torch.bfloat16) # Load into lower precision...
gc.collect()
torch.cuda.empty_cache()

#%% # Takes 10 seconds

model = model.cuda()
gc.collect()
torch.cuda.empty_cache()

# %%

if False:
    # Sanity check that we can run forward passes 
    tokenizer = model.tokenizer if TL else transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

my_sentence = """Successorship behavior: the successor head pushes for the successor of a token in the context. We
say this behavior occurs when one of the top-5-attended tokens is in the successorship dataset, and
the correct next token is the successor of t.
Acronym behavior: the successor head pushes for an acronym of words in the context. We say this
behavior occurs when the correct next token is an acronym whose last letter corresponds to the first
letter of the top-1-attended token. (For example, if the successor head attends most strongly to the
token ‘Defense’, and the correct next token is ‘OSD’.)
Copying behavior: the successor head pushes for a previous token in the context. We say this
behavior occurs when the correct next token t has already occurred in the prompt, and token t is one
of the top-5-attended tokens.
Greater-than behavior: the successor head pushes for a token greater than a previous token in the
context. We say this behavior occurs when we do not observe successorship behavior, but when the
correct next token is still part of an ordinal sequence and has greater order than some top-5-attended
token (e.g. if the successor head attends to the token ‘first’ and the model predicts the token
‘third’.)"""

if False:
    my_tokens = tokenizer.encode(my_sentence, return_tensors="pt").cuda() # First loss is really high but whatever, it's really low on average
else:
    my_tokens = model.to_tokens(my_sentence).cuda()

#%%

if False:
    my_logits = hf_model(my_tokens).logits
else:
    my_logits = model(my_tokens)
my_logprobs = my_logits.log_softmax(dim=-1)
my_neglogprobs = -my_logprobs
my_loss = my_neglogprobs[0, torch.arange(my_neglogprobs.shape[1]-1), my_tokens[0][1:]]
print(my_loss.mean()) # Yeah pretty small... 2 for LLAMA-7B...

#%%

# Probably use the `streaming` option on this...
hf_dataset_name = "suolyer/pile_pile-cc"
# Dataset
import datasets
hf_iter_dataset = iter(datasets.load_dataset(hf_dataset_name, streaming=True)["validation"])

# %%

# Step 1 --- Run up to Layer L (end of Succ Head)
# Step 2 --- Calculate the Mean Ablated Head Output, etc
# Step 3 continue forward pass

#%%

def get_filtered_dataset(model, batch_size=30, seed: int = 1729, device="cuda", max_seq_len=1024):
    hf_dataset_name = "suolyer/pile_pile-cc" # Probably somewhere in LLAMA data
    hf_iter_dataset = iter(datasets.load_dataset(hf_dataset_name, streaming=True)["validation"])

    filtered_tokens = []
    targets = []  # targets for prediction

    print("Not rapid, but not THAT slow :-) ")
    _idx = -1
    while len(filtered_tokens) < batch_size:
        _idx += 1
        cur_tokens = model.to_tokens(next(hf_iter_dataset)["text"], truncate=False).tolist()[0]
        if (
            len(cur_tokens) > max_seq_len # Greater Than so that we have all the targets for the context!!!
        ):  # so we're not biasing towards early sequence positions...
            filtered_tokens.append(cur_tokens[:max_seq_len])
            targets.append(cur_tokens[1 : max_seq_len + 1])

    mybatch = torch.LongTensor(filtered_tokens).to(device)
    mytargets = torch.LongTensor(targets).to(device)
    return mybatch, mytargets

tokens, targets = get_filtered_dataset(model, batch_size=20, max_seq_len=1024)

# %%

import itertools

for layer, head in [(11,10)]:  # list(itertools.product([model.cfg.n_layers-1, model.cfg.n_layers-2], list(range(model.cfg.n_heads)))):
    print("HEAD!!!", layer, head)
    list_of_neg_cosine_sims = []

    import warnings
    warnings.warn("Not short tokens")

    output, cache = model.run_with_cache(
        tokens,
        names_filter=[
            # f"blocks.{layer}.attn.hook_rot_q",
            # f"blocks.{layer}.attn.hook_rot_k",
            f"blocks.{layer}.attn.hook_pattern",          # Reconstruct head output from these
            f"blocks.{layer}.attn.hook_z",       # Logit lens god I hope works
        ],
        stop_at_layer = layer + 1,
    )

    # Implement just "suppression" here, no filtering.

    z_for_head = cache[f"blocks.{layer}.attn.hook_z"][:, :, head]
    assert len(z_for_head.shape)==3 # batch, seq_len, dhead
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]
    import einops
    ov_per_position = einops.einsum(
        z_for_head,
        model.W_O[layer, head],
        "batch seq_len dhead, dhead d_model -> batch seq_len d_model",
    )

    # Delete the dumb shit (currently unused)
    dumb_shit = einops.einsum(
        ov_per_position,
        pattern.to(ov_per_position.dtype), # Downcast
        "batch key_index d_model, batch query_index key_index -> batch query_index d_model",
    )
    # clever_shit = project(
    #     ov_per_position, # Shape batch key_index d_model
    #     model.W_U.T[short_tokens], # batch key_index d_model
    # )
    # Well that sucks

    unnormed_directions = model.W_U.T[tokens]
    normed_directions = unnormed_directions / unnormed_directions.norm(dim=-1, keepdim=True)
    projection_sizes = einops.einsum(
        ov_per_position,    
        normed_directions,
        "batch query_index d_model, batch key_index d_model -> batch query_index key_index"
    )
    # projection_sizes = einshape( # Ugh no einsum
    #     "bkd,bkd->bk1",
    #     ov_per_position,    
    #     normed_directions,
    # )

    # projection = projection_sizes * normed_directions
    norms = ov_per_position.norm(dim=-1)

    # Okay just replace at end of model???
    # Or just see the sizes of these damn thing
    # Do like: where attention is >0.5 and not BOS, check how big the projections are

    positions = (pattern>0.5) & (torch.arange(pattern.shape[-1])[None, None] > 0).to(pattern.device)

    for b, q, k in positions.nonzero(): # Still bugged
        cur_projection = projection_sizes[b, q, k]
        norm = norms[b, q]
        print(cur_projection / norm)
        list_of_neg_cosine_sims.append(-(cur_projection / norm).item())

    gc.collect()
    torch.cuda.empty_cache() # Is this actually big???

# %%

hist(
    [list_of_neg_cosine_sims],
    # Use percentage
    histnorm='probability',
    return_fig=True,
).update_layout(
    title=f"Histogram of Cosine Similarities between Head Output and Negative W_U on max attention token;\nModel: {model.cfg.model_name}; Head: {head}; Layer: {layer}",
    xaxis_title="Cosine Similarity",
    yaxis_title="Density",
    width=1200,
    height=600,
    # Label for bars
    # barmode="overlay",
    # bargap=0.1,
    # bargroupgap=0.1,
    showlegend=False,
).show()