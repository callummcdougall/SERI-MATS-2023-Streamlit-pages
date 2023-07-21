# %% [markdown] [1]:

"""
Mixing key_and_query_projection and arthur_signal_owt here
"""

from transformer_lens.cautils.ioi_dataset import _logits_to_ave_logit_diff
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
    seed=35795,
    device="cuda",
)

#%%

def editor_hook(z, hook, new_value, head_idx=7, replace=False):
    """Set head_idx None to edit the whole of z"""

    if "normalized" in hook.name:
        print("Ensuring this is Post LN")
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
ln_pre_name = f"blocks.10.ln1.hook_scale"
ln_final_name = f"ln_final.hook_scale"

model.set_use_attn_result(True)
logits, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name in [hook_pre_name, "blocks.9.attn.hook_result", "blocks.10.attn.hook_result", ln_pre_name, ln_final_name],
)

pre_neg_resid = cache[hook_pre_name][torch.arange(N), ioi_dataset.word_idx["end"]]
pre_neg_scale = cache[ln_pre_name][torch.arange(N), ioi_dataset.word_idx["end"], 7]
end_scale = cache[ln_final_name][torch.arange(N), ioi_dataset.word_idx["end"]]

vanilla_logit_diff = _logits_to_ave_logit_diff(logits, ioi_dataset)
print(vanilla_logit_diff.item(), "is the vanilla logit diff")

# %%

results = {}
projector_heads = [(9, 6), (9, 9)]
FREEZE_INPUT_LAYER_NORM = True

for mode in ["parallel", "perp"]:
    thing_to_remove = torch.zeros((N, model.cfg.d_model)).to("cuda")
    for projector_head_layer, projector_head_idx in projector_heads:
        parallel_component, perp_component = project(
            cache[f"blocks.{projector_head_layer}.attn.hook_result"][torch.arange(N), ioi_dataset.word_idx["end"], projector_head_idx],
            model.W_U[:, ioi_dataset.io_tokenIDs].T,
        )

        thing_to_remove += (perp_component if mode == "parallel" else parallel_component) / (pre_neg_scale if FREEZE_INPUT_LAYER_NORM else 1.0)

    model.reset_hooks()
    model.add_hook(
        f"blocks.10.hook_q_{'normalized_' if FREEZE_INPUT_LAYER_NORM else ''}input",
        partial(editor_hook, new_value=-thing_to_remove, replace=False),
    )
    model.add_hook(
        ln_final_name,
        partial(editor_hook, new_value=end_scale, replace=True, head_idx=None),
    )
    pattern_name = "blocks.10.attn.hook_attn_scores"
    new_logits, new_cache = model.run_with_cache(ioi_dataset.toks, names_filter=lambda name: name==pattern_name)
    for token in ["IO", "S1", "BOS"]:
        if token == "BOS":
            token_idx = 0
        else:
            token_idx = ioi_dataset.word_idx[token]

        print(
            f"Average {token=} attention: {str(new_cache[pattern_name][torch.arange(N), 7, ioi_dataset.word_idx['end'], token_idx].mean().item())}",
        )
    new_logit_diff = _logits_to_ave_logit_diff(new_logits, ioi_dataset)
    print("While only keeping the", mode, "direction, we get logit diff", new_logit_diff.item())

#%%

# %%

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