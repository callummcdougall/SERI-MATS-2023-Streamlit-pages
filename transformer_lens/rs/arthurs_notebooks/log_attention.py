# %% [markdown] [4]:

"""
Copy of direct effect survey
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_metric_from_end_state, dot_with_query
import argparse

model: HookedTransformer = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)
model.set_use_attn_in(True)
DEVICE = "cuda"
SHOW_PLOT = True
DATASET_SIZE = 500
BATCH_SIZE = 20 # seems to be about the limit of what this box can handle
NUM_THINGS = 300
USE_RANDOM_SAMPLE = False
INDIRECT = True # disable for orig funcitonality
USE_GPT2XL = False

# %%

dataset = get_webtext(seed=17279)
max_seq_len = model.tokenizer.model_max_length

# %%

filtered_tokens = []
targets = []  # targets for prediction

print("Not rapid, but not THAT slow :-) ")
_idx = -1
while len(filtered_tokens) < DATASET_SIZE:
    _idx += 1
    cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
    if (
        len(cur_tokens) > max_seq_len
    ):  # so we're not biasing towards early sequence positions...
        filtered_tokens.append(cur_tokens[:max_seq_len])
        targets.append(cur_tokens[1 : max_seq_len + 1])

mybatch = torch.LongTensor(filtered_tokens[:BATCH_SIZE])
mytargets = torch.LongTensor(targets[:BATCH_SIZE])

#%%

logits, cache = model.run_with_cache(
    mybatch.to(DEVICE), 
    names_filter = lambda name: name in ["blocks.10.hook_resid_pre", "blocks.10.attn.hook_attn_scores"],
)
attn_score = cache["blocks.10.attn.hook_attn_scores"][:, 7]

#%%

normalized_query = cache["blocks.10.hook_resid_pre"]
normalized_query /= (normalized_query.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)

#%%

xs = []
ys = []
ys2 = []

for batch_idx in tqdm(range(BATCH_SIZE)):
    warnings.warn("Trimming to size 30")
    for seq_idx in range(30):
        denom = torch.exp(attn_score[batch_idx, seq_idx, :seq_idx+1]).sum()
        cur_seq_idx = 0
        for cur_seq_idx in range(seq_idx):

            outputs = dot_with_query(
                unnormalized_keys = cache["blocks.10.hook_resid_pre"][batch_idx, cur_seq_idx:cur_seq_idx+1],
                unnormalized_queries = normalized_query[batch_idx, seq_idx:seq_idx+1],
                model=model,
                layer_idx=10,
                head_idx=7,
                add_key_bias = True, 
                add_query_bias = True,
                normalize_keys = True,
                normalize_queries = False,
                use_tqdm=False,
            )

            query = normalized_query[batch_idx, seq_idx:seq_idx+1]
            parallel, perp = project(
                query,
                model.W_U.T[mybatch[batch_idx, cur_seq_idx:cur_seq_idx+1]],
            )

            para_score = dot_with_query(
                unnormalized_keys = cache["blocks.10.hook_resid_pre"][batch_idx, cur_seq_idx:cur_seq_idx+1],
                unnormalized_queries = parallel,
                model=model,
                layer_idx=10,
                head_idx=7,
                add_key_bias = True, 
                add_query_bias = True,
                normalize_keys = True,
                normalize_queries = False,
                use_tqdm=False,
            )
            perp_score = dot_with_query(
                unnormalized_keys = cache["blocks.10.hook_resid_pre"][batch_idx, cur_seq_idx:cur_seq_idx+1],
                unnormalized_queries = perp,
                model=model,
                layer_idx=10,
                head_idx=7,
                add_key_bias = True, 
                add_query_bias = True,
                normalize_keys = True,
                normalize_queries = False,
                use_tqdm=False,
            )

            xs.append(para_score.item()/denom.item())
            ys.append(perp_score.item()/denom.item())
            ys2.append((outputs.item(), denom))
# 
        # print(ys2)
        # if cur_seq_idx > 30:
        #     assert False

fig = px.scatter(
    x=xs, 
    y=ys,
)

fig.update_layout(
    xaxis_title="Log attention score from unembedding parallel projection",
    yaxis_title="Log attention score from unembedding perpendicular projection",
)

fig.show()

# %%
