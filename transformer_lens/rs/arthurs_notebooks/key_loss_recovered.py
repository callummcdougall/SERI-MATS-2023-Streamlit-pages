# %% [markdown] [1]:

"""
Mixing key_and_query_projection and arthur_signal_owt here
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
from transformer_lens.rs.callum.keys_fixed import (
    project,
)
from transformer_lens.rs.callum2.utils import ( 
    get_effective_embedding,
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
    model, batch_size=BATCH_SIZE, seed=17717, device="cuda", max_seq_len=MAX_SEQ_LEN
)
effective_embeddings = get_effective_embedding(model, use_codys_without_attention_changes=False)

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
    or name == f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_attn_scores"
    or name == f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_pattern"
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

original_end_state = cache[END_STATE_HOOK].clone()
batched_tokens_loss = get_metric_from_end_state(
    model=model,
    end_state=original_end_state,
    targets=targets,
)

# %%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX].clone()
assert head_output.shape == (BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model)

# %%

mean_head_output = einops.reduce(head_output, "b s d -> d", reduction="mean")

# %%

mean_ablated_end_states = (
    cache[get_act_name("resid_post", model.cfg.n_layers - 1)].clone()
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

top5p_batch_indices = torch.LongTensor([x[0] for x in top_5_percent])
top5p_seq_indices = torch.LongTensor([x[1] for x in top_5_percent])

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
    + [f"blocks.{layer_idx}.hook_mlp_out" for layer_idx in range(NEGATIVE_LAYER_IDX)]
    + [f"blocks.{layer_idx}.attn.hook_result" for layer_idx in range(NEGATIVE_LAYER_IDX)]
    + [f"bias.{layer_idx}" for layer_idx in range(NEGATIVE_LAYER_IDX)]
): # all the writing weights
    if "bias" in hook_name:
        layer_idx = int(hook_name.split(".")[1])
        all_residual_stream[hook_name] = einops.repeat(model.b_O[layer_idx], "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN).clone()
        # all_residual_stream[hook_name + ".mlp"] = einops.repeat(model.blocks[layer_idx].mlp.b_out, "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN)
    elif "attn" in hook_name:
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

pre_negative_residual_state = cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)].clone()

norm_one = pre_negative_residual_state.norm().item()
norm_two = sum(list(all_residual_stream.values())).norm().item()
assert abs(norm_one - norm_two) < 1e-2, (norm_one, norm_two)

#%%

negative_head_layer_norm_scale = (1/np.sqrt(model.cfg.d_model)) * (pre_negative_residual_state.norm(dim=-1, keepdim=True)) #.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5) # eh? seems wrong but ok

assert abs((pre_negative_residual_state/negative_head_layer_norm_scale).norm(dim=-1) - np.sqrt(model.cfg.d_model)).norm() < 1e-2, (pre_negative_residual_state/negative_head_layer_norm_scale).norm(dim=-1)

top5p_negative_head_layer_norm_scale = negative_head_layer_norm_scale[top5p_batch_indices]

#%%

top5p_keys_in_normalized = {k: v[top5p_batch_indices] / top5p_negative_head_layer_norm_scale for k, v in all_residual_stream.items()}

#%%

resid_sum = sum(list(top5p_keys_in_normalized.values()))
assert (resid_sum.norm(dim=-1) - np.sqrt(model.cfg.d_model)).norm().item() < 1e-1

#%%

bos_key_in = cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][:, 0]
assert (bos_key_in - bos_key_in[0:1]).norm().item() < 1e-2
bos_key_in = bos_key_in[0]

top5p_query_in = pre_negative_residual_state[top5p_batch_indices, top5p_seq_indices]

#%%

model_attention_scores = cache[f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_attn_scores"][:, NEGATIVE_HEAD_IDX]
model_attention_scores_diagonal = model_attention_scores[:, torch.arange(model_attention_scores.shape[1]), torch.arange(model_attention_scores.shape[2])]

normal_attention_scores = dot_with_query(
    unnormalized_keys=pre_negative_residual_state.clone(),
    unnormalized_queries=pre_negative_residual_state.clone(),
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=True,
    add_query_bias=True,
    normalize_keys=True,
    normalize_queries=True,
    use_tqdm=True,
)

# %%

torch.testing.assert_allclose(
    normal_attention_scores.cuda(),
    model_attention_scores_diagonal,
    atol=1e-2,
    rtol=1e-2,
)

#%%

attention_score_components = {}

# for attention_score_component_name in all_residual_stream.keys():    
#     if "attn" in attention_score_component_name:
#         hook_name = "_".join(attention_score_component_name.split("_")[:-1])
#         head_idx = int(attention_score_component_name.split("_")[-1])

attention_score_components = dot_with_query(
    unnormalized_keys=torch.stack(list(top5p_keys_in_normalized.values()), dim=0).clone(),
    unnormalized_queries=einops.repeat(top5p_query_in, "b d -> c b s d", s=MAX_SEQ_LEN, c=len(top5p_keys_in_normalized)).clone(),
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

attention_score_key_component = einops.repeat(dot_with_query(
    unnormalized_keys=torch.zeros((top5p_query_in.shape[0], model.cfg.d_model)).cuda(),
    unnormalized_queries=top5p_query_in.clone(),
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,   
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=True, # look!!!
    add_query_bias=True,
    normalize_keys=False,   
    normalize_queries=True,
    use_tqdm=True,
), "b -> b s", s=MAX_SEQ_LEN)

#%%

total_attention_score = attention_score_components.sum(dim=0) + attention_score_key_component

#%%

manually_created_attention_scores = torch.zeros((BATCH_SIZE, MAX_SEQ_LEN)).cuda()

for batch_idx in range(BATCH_SIZE):
    for seq_idx in range(MAX_SEQ_LEN):

        if top5p_seq_indices[batch_idx] < seq_idx:
            continue

        manually_created_attention_scores[batch_idx, seq_idx] = cache[f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_attn_scores"][top5p_batch_indices[batch_idx], NEGATIVE_HEAD_IDX, top5p_seq_indices[batch_idx], seq_idx]

        assert abs(total_attention_score[batch_idx, seq_idx].item()- manually_created_attention_scores[batch_idx, seq_idx].item()) < 5e-1, (total_attention_score[batch_idx, seq_idx].item(), manually_created_attention_scores[batch_idx, seq_idx].item())

# %%

top5p_bos_attention = cache[utils.get_act_name("attn_scores", NEGATIVE_LAYER_IDX)][top5p_batch_indices, NEGATIVE_HEAD_IDX, top5p_seq_indices, 0]
top5p_max_non_bos_attention = cache[utils.get_act_name("attn_scores", NEGATIVE_LAYER_IDX)][top5p_batch_indices, NEGATIVE_HEAD_IDX, top5p_seq_indices, 1:].max(dim=-1)
top5p_max_non_bos_attention_indices = top5p_max_non_bos_attention.indices + 1
top5p_max_non_bos_attention_values = top5p_max_non_bos_attention.values

go.Figure(
    data = [
        go.Bar(x = list(range(BATCH_SIZE)), y = top5p_bos_attention.cpu().numpy(), name="BOS"),
        go.Bar(x = list(range(BATCH_SIZE)), y = top5p_max_non_bos_attention_values.cpu().numpy(), name="Max non-BOS"),
    ]
).show()
        
fig = px.scatter(
    x = top5p_seq_indices.cpu().numpy(),
    y = top5p_max_non_bos_attention_indices.cpu().numpy(),
)
# add y=x line
fig.add_trace(
    go.Scatter(
        x = list(range(MAX_SEQ_LEN)),
        y = list(range(MAX_SEQ_LEN))
    )
)
fig.show() # all seems good bro to do attention scores on max - BOS : ) 

#%%

bos_contribution_components = {
    k: v[0, 0].clone() for k, v in all_residual_stream.items()
}
torch.testing.assert_close(
    sum(list(bos_contribution_components.values())),
    bos_key_in,
)

bos_key_in_normalization = bos_key_in.var().pow(0.5) + model.cfg.eps
normalized_bos_contribution_components = {k: v / bos_key_in_normalization for k, v in bos_contribution_components.items()}

#%%

bos_attention_score_components = dot_with_query(
    unnormalized_keys=einops.repeat(torch.stack(list(normalized_bos_contribution_components.values()), dim=0), "c d -> c b d", b=BATCH_SIZE).clone(),
    unnormalized_queries=einops.repeat(top5p_query_in, "b d -> c b d", c=len(bos_contribution_components)).clone(),
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=False,
    add_query_bias=True,
    normalize_keys=False,
    normalize_queries=True,
    use_tqdm=True,
)

bos_attention_bias = dot_with_query(
    unnormalized_keys=torch.zeros((BATCH_SIZE, model.cfg.d_model)).cuda(),
    unnormalized_queries=top5p_query_in,
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=True,
    add_query_bias=True,
    normalize_keys=False,
    normalize_queries=True,
    use_tqdm=True,
)

# %%

top5p_key_attention_max_in = {
    k: v[torch.arange(len(v)), top5p_max_non_bos_attention_indices].clone() for k, v in top5p_keys_in_normalized.items()
}

#%%

top5p_ka = sum(list(top5p_key_attention_max_in.values()))

torch.testing.assert_allclose(
    top5p_ka.norm(dim=-1),
    np.sqrt(model.cfg.d_model) * torch.ones((BATCH_SIZE,)).cuda(),
    atol=1e-2,
    rtol=1e-2,
)

torch.testing.assert_allclose(
    top5p_ka,
    pre_negative_residual_state[top5p_batch_indices, top5p_max_non_bos_attention_indices] * np.sqrt(model.cfg.d_model) / pre_negative_residual_state[top5p_batch_indices, top5p_max_non_bos_attention_indices].norm(dim=-1, keepdim=True),
)

#%%

attention_score_to_max = dot_with_query(
    unnormalized_keys=torch.stack(list(top5p_key_attention_max_in.values()), dim=0).clone(),
    unnormalized_queries=einops.repeat(top5p_query_in, "b d -> c b d", c=len(top5p_key_attention_max_in)).clone(),
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=False,
    add_query_bias=True,
    normalize_keys=False,
    normalize_queries=True,
    use_tqdm=True,
)

attention_score_to_max_bias = dot_with_query(
    unnormalized_keys=torch.zeros((BATCH_SIZE, model.cfg.d_model)).cuda(),
    unnormalized_queries=top5p_query_in.clone(),
    model=model,
    layer_idx=NEGATIVE_LAYER_IDX,
    head_idx=NEGATIVE_HEAD_IDX,
    add_key_bias=True,
    add_query_bias=True,
    normalize_keys=False,
    normalize_queries=True,
    use_tqdm=True,
)

# %%

difference_in_scores = attention_score_to_max - bos_attention_score_components
correct_difference_in_scores =  top5p_max_non_bos_attention_values.cpu() - top5p_bos_attention.cpu()

px.scatter(
    x = bos_attention_score_components.cpu().sum(dim=0) + bos_attention_bias.cpu(),
    y = top5p_bos_attention.cpu().numpy(),
).show()

px.scatter(
    x = attention_score_to_max.cpu().sum(dim=0) + attention_score_to_max_bias.cpu(),
    y = top5p_max_non_bos_attention_values.cpu().numpy(),
).show()    

px.scatter(
    x = difference_in_scores.cpu().sum(dim=0),
    y = correct_difference_in_scores.cpu().numpy(),
).show()

torch.testing.assert_allclose(
    difference_in_scores.cpu().sum(dim=0),
    correct_difference_in_scores,
    atol=1e-2,
    rtol=1e-2,
)

# go.Figure(
#     data = [
#         go.Bar(x = list(range(BATCH_SIZE)), y = top5p_bos_attention.cpu().numpy(), name="BOS"),
#         go.Bar(x = list(range(BATCH_SIZE)), y = top5p_max_non_bos_attention_values.cpu().numpy(), name="Max non-BOS"),
#     ]
# ).show()

# %%

for thing_name, thing in zip(["score on max", "bos", "difference between max and BOS"], [attention_score_to_max, bos_attention_score_components, difference_in_scores], strict=True):
    print(thing.sum())
    fig=px.bar(
        x = top5p_key_attention_max_in.keys(),
        y = thing.mean(dim=-1),
        error_y = thing.std(dim=-1),
    )
    fig.update_layout(
        title="Mean attention score to " + thing_name,
        # yaxis = dict(range=[0, 1]),
    )
    fig.show()
# %%
