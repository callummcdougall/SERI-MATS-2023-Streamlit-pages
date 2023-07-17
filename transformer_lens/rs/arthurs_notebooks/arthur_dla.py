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

model = HookedTransformer.from_pretrained("gpt2")
NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]

#%%

mybatch, mytargets = get_filtered_webtext()

#%%

model.set_use_attn_result(True)
head_activations = model.run_with_cache(
    mybatch,
    names_filter = lambda name: name == get_act_name("result", NEGATIVE_LAYER_IDX)
)[1][get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]

mean_activations = head_activations.mean(dim=(0, 1))

#%%

for subtract_mean in [False, True]:
    current_activations = head_activations - (mean_activations if subtract_mean else 0.0)

    current_attributions = einops.einsum(
        current_activations,
        model.W_U,
        "batch seq_len d_model, d_model d_vocab -> batch seq_len d_vocab",
    )

    K = 5
    top_attributions = torch.topk(
        current_attributions.abs(),
        k=K,
        dim=-1,
    ).indices
    assert list(top_attributions.shape ) == [1, K]

# %%
