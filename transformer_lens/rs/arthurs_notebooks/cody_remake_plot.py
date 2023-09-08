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

#%%

# Create a DataFrame to hold the data
MODES = ['parallel', 'perp']
df = pd.DataFrame()


fig = go.Figure()

for mode in MODES:
    x = deepcopy(x_data[mode]) # Replace with your logit_diffs_per_head data
    y = deepcopy(y_data[mode])  # Replace with your head_logit_diffs data
    text = deepcopy(text_data[mode])
    temp_df = pd.DataFrame({
        'x_data': x,
        'y_data': y,
        'text_data': text,
        'mode': mode,
    })

    textpositions = ['top right' if (mode == 'parallel') == (idx%2 == 1) else 'bottom right' for idx in range(len(x))]
    for idx in range(len(x)):
        if text[idx] in ['10.6', '10.2']:
            textpositions[idx] = "middle left"

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers+text',
            text=text,
            textposition=textpositions,
            marker=dict(size=5, color='red' if mode == 'parallel' else 'blue'),
            # Use linebreak after IO
            name=f"""Only include Name Moving Heads' IO<br>{'perpendicular' if mode=='perp' else 'parallel'} directions in the query""",
        )
    )


fig.update_layout(
    paper_bgcolor='rgba(255,255,255,255)',
    plot_bgcolor='rgba(255,255,255,255)'
)
# fig.update_traces(textposition='top center')
fig.update_yaxes(showgrid=True)
fig.update_xaxes(showgrid=True)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, gridwidth=1, gridcolor='gray')
fig.update_layout(
    shapes=[
        dict(
            type='line',
            x0=0,
            x1=1,
            xref='paper',
            y0=0,
            y1=0,
            yref='y',
            line=dict(color='black', width=2)
        ),
        dict(
            type='line',
            x0=0,
            x1=0,
            xref='x',
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='black', width=2)
        )
    ]
)
# Add a y=x line
fig.add_trace(
    go.Scatter(
        x=[-2.5, 1],
        y=[-2.5, 1],
        mode='lines',
        name='y=x',
        line=dict(color='black', width=2, dash='dash')
    )
)

# Add y axis label
fig.update_layout(
    yaxis_title="Post-intervention logit difference",
    xaxis_title="Pre-intervention logit difference",
    title = "Self-repairing attention heads under projection interventions",
)

fig.show()

#%%

