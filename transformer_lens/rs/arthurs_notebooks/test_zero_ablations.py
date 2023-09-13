#%%

from transformer_lens.cautils.notebook import *

# %%

model = transformer_lens.HookedTransformer.from_pretrained("gpt-neo-125M")

# %%

from transformer_lens.rs.arthurs_notebooks.arthurs_utils import get_filtered_webtext

my_batch, my_targets = get_filtered_webtext(model, batch_size=30, seed= 1729, device="cuda", max_seq_len=1024, dataset="NeelNanda/pile-10k")

# %%


# %%

def compute_loss(model):
    predictions = model(
        my_batch[:10],
    )
    logprobs = predictions.log_softmax(dim=-1)
    neglogprobs = -logprobs[torch.arange(len(predictions)).unsqueeze(1), torch.arange(predictions.shape[1]).unsqueeze(0), my_targets[:10]]
    return neglogprobs.mean()

# %%

model.set_use_attn_result(True)

for head in [2, 3, 5]:
    model.reset_hooks()
    def zer(x, hook):
        x[:, :, head] = 0.0
        return x
    model.add_hook(
        f"blocks.0.attn.hook_result",
        zer,
    )
    loss = compute_loss(model)
    print(loss)
# %%
