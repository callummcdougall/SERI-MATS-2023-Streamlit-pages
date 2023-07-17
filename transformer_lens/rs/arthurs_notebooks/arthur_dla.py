#%%

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
from transformer_lens.loading_from_pretrained import MODEL_ALIASES as MA
import argparse

# NOT ACTUALLY USED
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default=None)

if ipython is not None:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

# %%

model = HookedTransformer.from_pretrained("gpt2", device="cpu")
NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
# NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = 9, 9

#%%

mybatch, mytargets = get_filtered_webtext(model=model, device="cpu")

#%%

model.set_use_attn_result(True)
head_activations = model.run_with_cache(
    mybatch,
    names_filter = lambda name: name == get_act_name("result", NEGATIVE_LAYER_IDX)
)[1][get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
mean_activations = head_activations.mean(dim=(0, 1))
print(mean_activations.norm())

#%%

K = 5

for subtract_mean in [True, False]:
    current_activations = head_activations.clone() - (mean_activations.clone() if subtract_mean else 0.0)
    print(current_activations.norm())
    current_activations_layer_normed = (current_activations / current_activations.norm(dim=-1, keepdim=True)) * np.sqrt(model.cfg.d_model)

    print(current_activations_layer_normed.norm(), model.W_U.norm())

    current_attributions = einops.einsum(
        current_activations_layer_normed,
        model.W_U,
        "batch seq_len d_model, d_model d_vocab -> batch seq_len d_vocab",
    )

    top_attribution_indices = torch.topk(
        current_attributions.abs(),
        k=K,
        dim=-1,
    ).indices

    assert list(top_attribution_indices.shape) == [mybatch.shape[0], mybatch.shape[1], K]

    top_attribution_values = current_attributions[ # i hate gather
        torch.arange(mybatch.shape[0]).unsqueeze(-1).unsqueeze(-1),
        torch.arange(mybatch.shape[1]).unsqueeze(-1).unsqueeze(0),
        top_attribution_indices,
    ]

    print("Proportion of positive logit attributions:", ((torch.nn.functional.relu(top_attribution_values) > 0).int().sum() / top_attribution_values.numel()).item())

# %%
