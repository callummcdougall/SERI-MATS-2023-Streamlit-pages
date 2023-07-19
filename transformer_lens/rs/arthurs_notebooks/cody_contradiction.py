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

def editor_hook(z, hook, new_value, head_idx=7):
    if "normalized" in hook.name:
        print("Ensuring this is Post LN")
        assert (z.norm(dim=-1) - np.sqrt(model.cfg.d_model)).norm().item() < 1e-3
    z[torch.arange(N), ioi_dataset.word_idx["end"], head_idx, :] = new_value
    return z

#%%

hook_pre_name = f"blocks.10.hook_resid_pre"
logits, cache = model.run_with_cache(
    ioi_dataset.toks,
    names_filter = lambda name: name == hook_pre_name
)
pre_neg_resid = cache[hook_pre_name][torch.arange(N), ioi_dataset.word_idx["end"]]

# %%

results = {}

pre_neg_resid_scaled = pre_neg_resid.clone()
pre_neg_resid_scaled /= pre_neg_resid_scaled.norm(dim=-1, keepdim=True)
pre_neg_resid_scaled *= np.sqrt(model.cfg.d_model)

for FREEZE_LN in [True, False]:
    parallel_component, perp_component = project(
        pre_neg_resid_scaled if FREEZE_LN else pre_neg_resid,
        model.W_U[:, ioi_dataset.io_tokenIDs].T,
    )

    model.set_use_split_qkv_input(True)
    model.set_use_split_qkv_normalized_input(True)

    vanilla_logit_diff = _logits_to_ave_logit_diff(logits, ioi_dataset)

    for new_name, new_value in [("parallel", parallel_component), ("perp", perp_component)]:
        model.reset_hooks()
        model.add_hook(
            "blocks.10.hook_q_normalized_input" if FREEZE_LN else "blocks.10.hook_q_input",
            partial(editor_hook, new_value=new_value),
        )

        new_logits = model(ioi_dataset.toks)
        new_logit_diff = _logits_to_ave_logit_diff(new_logits, ioi_dataset)

        print("Change in logit diff", new_logit_diff - vanilla_logit_diff)

        model.reset_hooks()

        results[(FREEZE_LN, new_name)] = (new_logit_diff).item()

#%%

dirs = ["parallel", "perp"]

fig = go.Figure(data=[
    go.Bar(name=f'{FREEZE_LN=}', x=[f"{FREEZE_LN=}, {dir=}" for dir in dirs], y=[results[(FREEZE_LN, dir)] for dir in dirs])
    for FREEZE_LN in [True, False]
] + [go.Bar(name="vanilla", x=[f"Normal model"], y=[vanilla_logit_diff.item()])])
fig.update_layout(barmode='group', title=f"Logit difference for {N} samples. `Dir` means we keep only this component")

# make x title
fig.update_xaxes(title_text="Freeze LN, Dir")
fig.update_yaxes(title_text="Logit difference")

fig.show()

# %%

# side quest: cosine similarities between pos embeddings of small

cosine_sims = torch.zeros((model.cfg.n_ctx, model.cfg.n_ctx))

for i in range(model.cfg.n_ctx):
    for j in range(model.cfg.n_ctx):
        cosine_sims[i, j] = torch.cosine_similarity(model.pos_embed.W_pos[i], model.pos_embed.W_pos[j], dim=0)

#%%

imshow(
    cosine_sims,
    title="Cosine similarity between position embeddings",
    # xlabel="Position",
    # ylabel="Position",
)
# %%

webtext = get_webtext()

# %%

example=webtext[0].split(" ")

words = []
vals = []

SPACE_MODE = False

for word in example:
    try:
        lower_word = " "+word.lower()
        lower_token = model.to_single_token(lower_word)

        if SPACE_MODE:
            upper_token = model.to_single_token(lower_word[1:])
            assert word not in words

        else:
            upper_word = " "+word[:1].upper() + word[1:].lower()
            upper_token = model.to_single_token(upper_word)
            assert word not in words
            assert upper_token != lower_token

    except:
        pass
    else:
        vals.append(torch.cosine_similarity(model.W_U.T[lower_token], model.W_U.T[upper_token], dim=0).item())
        words.append(word)


# %%

# sort this 
words, vals = zip(*sorted(zip(words, vals), key=lambda x: x[1]))

px.bar(
    x=words,
    y=vals,
).show()
# %%
