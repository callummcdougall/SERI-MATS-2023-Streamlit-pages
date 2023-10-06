#%%

from transformer_lens.cautils.notebook import * 
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *

#%%

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

# %%

model = HookedTransformer.from_pretrained_no_processing("gpt2")

# #%%

# gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

# %%

NUM_DOCS=3000
toks, targets = get_filtered_webtext(model, NUM_DOCS)

# %%

batch_size = 100
assert len(toks) % batch_size == 0
running_loss_average = 0.0
running_ee_average = 0.0

for batch_idx in tqdm(range(NUM_DOCS//batch_size)):
    start_idx = batch_idx * batch_size
    end_idx = (batch_idx+1) * batch_size
    model.reset_hooks()
    cur_toks = toks[start_idx:end_idx]
    
    # logits = gpt2(torch.tensor(cur_toks).to(gpt2.device))[0]
    logits = model(torch.tensor(cur_toks))

    neglogprobs = -logits.log_softmax(dim=-1)
    losses = neglogprobs.cpu()[torch.arange(neglogprobs.shape[0]).unsqueeze(1), torch.arange(neglogprobs.shape[1]).unsqueeze(0), targets[start_idx:end_idx].cpu()]
    running_loss_average = running_loss_average * ( batch_idx / (batch_idx+1) ) + losses.mean().item() / (batch_idx+1)

    del logits
    del neglogprobs
    del losses
    gc.collect()
    torch.cuda.empty_cache()

    model.reset_hooks()
    embeds = model.W_E[cur_toks]
    logits = model.run_with_hooks(
        cur_toks,
        fwd_hooks = [("blocks.0.hook_resid_post", lambda resid_post, hook: resid_post - embeds)],
    )
    # Get losses
    neglogprobs = -logits.log_softmax(dim=-1)
    losses = neglogprobs[torch.arange(neglogprobs.shape[0]).unsqueeze(1), torch.arange(neglogprobs.shape[1]).unsqueeze(0), targets[start_idx:end_idx]]
    running_ee_average = running_ee_average * ( batch_idx / (batch_idx+1) ) + losses.mean().item() / (batch_idx+1)

    print(running_loss_average, running_ee_average)

    del embeds
    del logits
    del neglogprobs
    del losses
    gc.collect()
    torch.cuda.empty_cache()

# %%
