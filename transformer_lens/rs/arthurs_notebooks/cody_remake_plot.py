# %% [markdown] [1]:

"""
Mixing key_and_query_projection and arthur_signal_owt here
Renamed to Cody plot because I want to demo a plot
"""

from transformer_lens.cautils.ioi_dataset import _logits_to_ave_logit_diff
from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
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

# %% [markdown] [2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_split_qkv_normalized_input(True)

# %%

N=100
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    seed=32,
    device="cuda",
)

#%%

def editor_hook(z, hook, new_value, head_idx=7, replace=False):
    """Set head_idx None to edit the whole of z"""

    if "normalized" in hook.name:
        assert (z.norm(dim=-1) - np.sqrt(model.cfg.d_model)).norm().item() < 1e-3

    if head_idx is not None:
        if replace:
            z[torch.arange(N), ioi_dataset.word_idx["end"], head_idx, :] = new_value
        else:
            z[torch.arange(N), ioi_dataset.word_idx["end"], head_idx, :] += new_value
    else:
        assert replace
        assert z[torch.arange(N), ioi_dataset.word_idx["end"]].shape == new_value.shape, (z.shape, new_value.shape)
        z[torch.arange(N), ioi_dataset.word_idx["end"]] = new_value

    return z

#%%

hook_pre_name = f"blocks.10.hook_resid_pre"
ln_pre10_name = f"blocks.10.ln1.hook_scale"
ln_pre11_name = f"blocks.11.ln1.hook_scale"
ln_final_name = f"ln_final.hook_scale"

model.set_use_attn_result(True)
logits, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name in [hook_pre_name] +  [f"blocks.{layer_idx}.attn.hook_result" for layer_idx in range(8, 12)] + [f"blocks.{layer_idx}.hook_resid_pre" for layer_idx in range(9, 12)] + ["blocks.10.attn.hook_result"] + [ln_pre10_name, ln_pre11_name, ln_final_name],
)

pre_neg_resid = cache[hook_pre_name][torch.arange(N), ioi_dataset.word_idx["end"]]
pre_scales = torch.zeros(12, 100, 1).cuda()
for layer, an_ln_pre_name in zip([10, 11], [ln_pre10_name, ln_pre11_name]):
    pre_scales[layer] = cache[an_ln_pre_name][torch.arange(N), ioi_dataset.word_idx["end"], 0]
end_scale = cache[ln_final_name][torch.arange(N), ioi_dataset.word_idx["end"]]

vanilla_logit_diff = _logits_to_ave_logit_diff(logits, ioi_dataset)
print(vanilla_logit_diff.item(), "is the vanilla logit diff")

#%%

logit_diff_directions = model.W_U.T[ioi_dataset.io_tokenIDs] - model.W_U.T[ioi_dataset.s_tokenIDs]

#%%

# Get the logit difference for each model component

logit_diffs_per_head = torch.zeros(12, 12)

for layer_idx in range(8, 12):
    for head_idx in range(12):
        # Get the logit difference for this head alone

        scaled_head_contribution = cache[f"blocks.{layer_idx}.attn.hook_result"][torch.arange(N), ioi_dataset.word_idx["end"], head_idx] / end_scale

        head_logit_diff = einops.einsum(
            scaled_head_contribution,
            logit_diff_directions,
            "batch d_model, batch d_model -> batch",
        )

        logit_diffs_per_head[layer_idx, head_idx] = head_logit_diff.mean().item()

#%%

imshow(
    logit_diffs_per_head,
)

#%%

name_mover_head_outputs = torch.zeros(12, 12, 100, model.cfg.d_model).cuda()

NMHs = [
    # (8, 0), 
    # (8, 6), 
    # (8, 10),
    # (8, 11),
    (9, 6), # Tons of extra NMHs
    (9, 9),
    # (9, 7), 
    # (9, 0),
    (10, 0),
    (10, 7),
    # (10, 2), 
    # (10, 6),
]

parallel_component, perp_component = torch.zeros((12, 12, 100, model.cfg.d_model)).cuda(), torch.zeros((12, 12, 100, model.cfg.d_model)).cuda()

for layer_idx, head_idx in NMHs:
    NMH = (layer_idx, head_idx)
    name_mover_head_outputs[NMH[0], NMH[1]] += cache[f"blocks.{NMH[0]}.attn.hook_result"][torch.arange(N), ioi_dataset.word_idx["end"], NMH[1]]

    cur_parallel_component, cur_perp_component = project(
        name_mover_head_outputs[layer_idx, head_idx],
        model.W_U[:, ioi_dataset.io_tokenIDs].T, # - model.W_U[:, ioi_dataset.s_tokenIDs].T,
    )
    parallel_component[layer_idx, head_idx] = cur_parallel_component
    perp_component[layer_idx, head_idx] = cur_perp_component
    print(cur_parallel_component.norm(dim=-1).mean().item(), cur_perp_component.norm(dim=-1).mean().item())

# %%

results = {}
FREEZE_INPUT_LAYER_NORM = True
MODES = ["parallel", "perp"]
head_logit_diffs = {
    mode: torch.zeros((12, 12)) for mode in MODES
}
PERSON_MODE: Literal["Cody", "Arthur"] = "Arthur"

for mode in MODES:
    for layer_idx, head_idx in tqdm(list(itertools.product(range(10, 12), range(12)))):
        model.reset_hooks()
        if PERSON_MODE == "Arthur":
            thing_to_remove = (perp_component if mode == "parallel" else parallel_component).clone()
            thing_to_remove = thing_to_remove[:layer_idx].sum(dim=(0, 1))
            thing_to_remove /= (pre_scales[layer_idx] if FREEZE_INPUT_LAYER_NORM else 1.0)
            model.add_hook(
                f"blocks.{layer_idx}.hook_q_{'normalized_' if FREEZE_INPUT_LAYER_NORM else ''}input",
                partial(editor_hook, new_value=-thing_to_remove, replace=False, head_idx=head_idx),
            )
        elif PERSON_MODE == "Cody":
            for nmh_layer_idx, nmh_head_idx in NMHs:
                
                if (layer_idx, head_idx) == (nmh_layer_idx, nmh_head_idx):
                    continue

                thing_to_keep = (perp_component if mode == "perp" else parallel_component).clone()[nmh_layer_idx, nmh_head_idx]
                model.add_hook(
                    f"blocks.{nmh_layer_idx}.attn.hook_result",
                    partial(editor_hook, new_value=thing_to_keep, head_idx=nmh_head_idx, replace=True),
                )
        else:
            raise ValueError(f"Unknown person mode {PERSON_MODE}")

        model.add_hook(
            ln_final_name,
            partial(editor_hook, new_value=end_scale, replace=True, head_idx=None),
        )

        new_logits, new_cache = model.run_with_cache(ioi_dataset.toks, names_filter=lambda name: name==f"blocks.{layer_idx}.attn.hook_result")
        new_scaled_head_output = new_cache[f"blocks.{layer_idx}.attn.hook_result"][torch.arange(N), ioi_dataset.word_idx["end"], head_idx] / end_scale
        new_head_logit_diff = einops.einsum(
            new_scaled_head_output,
            logit_diff_directions,
            "batch d_model, batch d_model -> batch",
        )
        head_logit_diffs[mode][layer_idx, head_idx] = new_head_logit_diff.mean().item()

#%%

fig = go.Figure()

colors = {
    "parallel": "red",
    "perp": "blue",
}

layer_heads = {
    10: [2, 10, 6, 7],
    11: [10, 2],
}

# Initialize empty lists to collect data for each mode
x_data = {mode: [] for mode in MODES}
y_data = {mode: [] for mode in MODES}
text_data = {mode: [] for mode in MODES}

for mode in MODES:
    for layer_idx in range(10, 12):
        x_data[mode].extend(logit_diffs_per_head[layer_idx][layer_heads[layer_idx]].tolist())
        y_data[mode].extend(head_logit_diffs[mode][layer_idx][layer_heads[layer_idx]].tolist())
        text_data[mode].extend([f"{layer_idx}.{head_idx}" for head_idx in layer_heads[layer_idx]])

# Increase font size
fig.update_layout(
    font=dict(
        size=14,
    )
)

# Add the traces for each mode
for mode in MODES:
    fig.add_trace(
        go.Scatter(
            x=x_data[mode],
            y=y_data[mode],
            mode="markers" + ("+text" if mode == "parallel" else ""),
            name=f"Only include Name Moving Heads' IO-{mode} directions in the query",
            text=text_data[mode],
            marker=dict(
                color=colors[mode],
            ),
        )
    )

# Add y=x line
fig.add_trace(
    go.Scatter(
        x=[-3, 3],
        y=[-3, 3],
        mode="lines",
        name="y=x",
        marker=dict(
            color="black",
        )
    )
)

fig.update_traces(textposition='top left')

fig.update_layout(
    title="Self-repairing attention heads under query interventions",
    xaxis_title="Clean logit difference",
    yaxis_title="Post-intervention logit difference",
)

fig.show()

#%%

pier_token = model.to_single_token(" pier")

names = []
cosine_sims = []

for i in tqdm(range(model.cfg.d_vocab)):
    names.append(model.to_single_str_token(i))
    cosine_sims.append(torch.cosine_similarity(model.W_U.T[i], model.W_U.T[pier_token], dim=0).item())

#%%

# sort this
names, cosine_sims = zip(*sorted(zip(names, cosine_sims), key=lambda x: x[1]))

px.bar(
    x=names[-30:],
    y=cosine_sims[-30:],
    title="Cosine similarity between W_U[:, pier] and W_U[:, x] : biggest cosine sims",
).show()

# %%