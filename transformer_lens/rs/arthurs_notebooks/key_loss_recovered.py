# %% [markdown] [1]:

"""
Mixing key_and_query_projection and arthur_signal_owt here
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
from transformer_lens.rs.callum.keys_fixed import (
    project,
    get_effective_embedding_2,
)
from transformer_lens.rs.callum.orthogonal_query_investigation import (
    decompose_attn_scores_full,
    create_fucking_massive_plot_1,
    create_fucking_massive_plot_2,
    token_to_qperp_projection,
    FakeIOIDataset,
)

clear_output()
USE_IOI = False
LAYER_IDX, HEAD_IDX = 10, 7

# %% [markdown] [2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
model.set_use_attn_result(True)

# %%

MAX_SEQ_LEN = 512
BATCH_SIZE = 30
batched_tokens, targets = get_filtered_webtext(
    model, batch_size=BATCH_SIZE, seed=1717, device="cuda", max_seq_len=MAX_SEQ_LEN
)
effective_embeddings = get_effective_embedding_2(model)

# %%

# Find the top 5% of things by importance

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = 10, 7
END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"

attention_pattern_hook_name = get_act_name("pattern", NEGATIVE_LAYER_IDX)
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name == get_act_name("resid_pre", 1)
    or name == f"blocks.{NEGATIVE_LAYER_IDX}.hook_resid_pre"
    or name == f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_result"
    or name == attention_pattern_hook_name
    or "mlp_out" in name 
    or "hook_result" in name
    or name.endswith("_embed")
)

logits, cache = model.run_with_cache(
    batched_tokens,
    names_filter=names_filter1,
)
gc.collect()
torch.cuda.empty_cache()

# %%

original_end_state = cache[END_STATE_HOOK]
batched_tokens_loss = get_metric_from_end_state(
    model=model,
    end_state=original_end_state,
    targets=targets,
)

# %%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
assert head_output.shape == (BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model)

# %%

mean_head_output = einops.reduce(head_output, "b s d -> d", reduction="mean")

# %%

mean_ablated_end_states = (
    cache[get_act_name("resid_post", model.cfg.n_layers - 1)]
    - head_output
    + einops.repeat(mean_head_output, "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN)
)
mean_ablated_loss = get_metric_from_end_state(
    model=model,
    end_state=mean_ablated_end_states,
    targets=targets,
)

# %%

max_importance_examples = sorted(
    [
        (
            batch_idx,
            seq_idx,
            (mean_ablated_loss - batched_tokens_loss)[batch_idx, seq_idx].item(),
        )
        for batch_idx, seq_idx in itertools.product(
            range(BATCH_SIZE), range(MAX_SEQ_LEN)
        )
    ],
    key=lambda x: x[2],
    reverse=True,
)

# %%

# Get the top 5% of things by importance
all_top_5_percent = max_importance_examples[: len(max_importance_examples) // 20]

np.random.seed(799)
# warnings.warn("No shuffle!!!")
np.random.shuffle(all_top_5_percent)
top_5_percent = all_top_5_percent[:BATCH_SIZE]

top5p_batch_indices = [x[0] for x in top_5_percent]
top5p_seq_indices = [x[1] for x in top_5_percent]

# %%

top5p_tokens = batched_tokens[top5p_batch_indices]
top5p_targets = torch.LongTensor(
    [
        targets[top5p_batch_idx, top5p_seq_idx]
        for top5p_batch_idx, top5p_seq_idx in zip(
            top5p_batch_indices, top5p_seq_indices
        )
    ]
)

# %%

top5p_losses = batched_tokens_loss[top5p_batch_indices, top5p_seq_indices]

# %%

# 1. Make an attention score calculator that splits by key contributor w/ assertions
# 2. Implement the bias subtraction (I could fold this into 1.)
# 3. Try and get a handle on which component matters here, maybe combine with the loss recovered work

# %%

all_residual_stream = {}

for hook_name in (
    ["hook_embed", "hook_pos_embed"]
    + [f"blocks.{layer_idx}.hook_mlp_out" for layer_idx in range(LAYER_IDX)]
    + [f"blocks.{layer_idx}.attn.hook_result" for layer_idx in range(LAYER_IDX)]
): # all the writing weights
    if "attn" in hook_name:
        for head_idx in range(model.cfg.n_heads):
            all_residual_stream[f"{hook_name}_{head_idx}"] = cache[hook_name][
                :,
                :,
                head_idx,
                :,
            ]
    else:
        all_residual_stream[hook_name] = cache[hook_name][
            :, :, :
        ]

# %%

pre_negative_residual_state = cache[get_act_name("resid_pre", LAYER_IDX)]
negative_head_layer_norm_scale = (pre_negative_residual_state.norm(dim=(1, 2), keepdim=True) * np.sqrt(model.cfg.d_model))
top5p_negative_head_layer_norm_scale = negative_head_layer_norm_scale[top5p_batch_indices]
top5p_key_in = {k: v[top5p_batch_indices] / top5p_negative_head_layer_norm_scale for k, v in all_residual_stream.items()}

top5p_query_in = pre_negative_residual_state[top5p_batch_indices]

#%%

attention_score_components = {}

# for attention_score_component_name in all_residual_stream.keys():    
#     if "attn" in attention_score_component_name:
#         hook_name = "_".join(attention_score_component_name.split("_")[:-1])
#         head_idx = int(attention_score_component_name.split("_")[-1])

attention_score_components = dot_with_query(
    unnormalized_keys=torch.stack(list(top5p_key_in.values()), dim=0),
    unnormalized_queries=einops.repeat(top5p_query_in, "b s d -> c b s d", c=len(top5p_key_in)).clone(),
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=False,
    add_query_bias=True,
    normalize_keys=False,
    normalize_queries=True,
    use_tqdm=True,
)

#%%

attention_score_key_component = dot_with_query(
    unnormalized_keys=einops.repeat(torch.zeros((model.cfg.d_model,)).cuda(), "d -> b s d", b=top5p_query_in.shape[0], s=top5p_query_in.shape[1]).clone(),
    unnormalized_queries=top5p_query_in,
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,   
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=True, # look!!!
    add_query_bias=True,
    normalize_keys=False,   
    normalize_queries=True,
    use_tqdm=True,
)

#%%

# def dot_with_query(
#     unnormalized_keys: Float[torch.Tensor, "batch d_model"],
#     unnormalized_queries: Float[torch.Tensor, "batch d_model"],
#     model,
#     layer_idx,
#     head_idx,
#     add_key_bias: bool = True, 
#     add_query_bias: bool = True,
#     normalize_keys: bool = True,
#     normalize_queries: bool = True,
#     use_tqdm: bool = True,
# ):

#%%
