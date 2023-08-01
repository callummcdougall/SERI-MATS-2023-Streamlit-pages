# %% [markdown] [4]:

"""
Copy of direct effect survey
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_metric_from_end_state, dot_with_query
from transformer_lens.rs.callum2.explore_prompts.model_results_3 import get_effective_embedding

model: HookedTransformer = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)

if ipython is None and False: # 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-idx", type=int)
    parser.add_argument("--head-idx", type=int)
    args = parser.parse_args()
    LAYER_IDX = args.layer_idx
    HEAD_IDX = args.head_idx

else:
    LAYER_IDX = 10
    HEAD_IDX = 7

model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)
model.set_use_attn_in(True)
DEVICE = "cuda"
SHOW_PLOT = True
DATASET_SIZE = 500
BATCH_SIZE = 30 

#%%

dataset = get_webtext(seed=17279)
max_seq_len = model.tokenizer.model_max_length

# %%

filtered_tokens = []
targets = []  # targets for prediction

print("Not rapid, but not THAT slow :-) ")
_idx = -1
while len(filtered_tokens) < DATASET_SIZE:
    _idx += 1
    cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
    if (
        len(cur_tokens) > max_seq_len
    ):  # so we're not biasing towards early sequence positions...
        filtered_tokens.append(cur_tokens[:max_seq_len])
        targets.append(cur_tokens[1 : max_seq_len + 1])

mybatch = torch.LongTensor(filtered_tokens[:BATCH_SIZE])
mytargets = torch.LongTensor(targets[:BATCH_SIZE])

#%%

logits, cache = model.run_with_cache(
    mybatch.to(DEVICE), 
    # names_filter = lambda name: name in [f"blocks.{LAYER_IDX}.hook_resid_pre", f"blocks.{LAYER_IDX}.attn.hook_attn_scores", f"blocks.{LAYER_IDX}.attn.hook_result", "blocks.11.hook_resid_post"],
    names_filter = lambda name: name.endswith(("hook_attn_scores", "hook_result", "hook_resid_pre", ".11.hook_resid_post")),
    device="cpu",
)

#%%

resid_pre = cache["blocks.10.hook_resid_pre"]
scaled_resid_pre = resid_pre / (resid_pre.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)

logit_lens = einops.einsum(
    scaled_resid_pre.to(DEVICE),
    model.W_U,
    "batch seq_len d_model, d_model d_vocab -> batch seq_len d_vocab",
)
logit_lens_topk = logit_lens.topk(5, dim=-1).indices

#%%

W_EE = get_effective_embedding(model)['W_E (including MLPs)']

# %%

W_EE_toks: Float[Tensor, "batch seqK d_model"] = W_EE[mybatch]

#%%

for LAYER_IDX, HEAD_IDX in [(10, 7)] +  list(itertools.product(range(9, 12), range(12))):
        attn_score = cache[f"blocks.{LAYER_IDX}.attn.hook_attn_scores"][:, HEAD_IDX].to(DEVICE)
        head_output = cache[f"blocks.{LAYER_IDX}.attn.hook_result"][:, :, HEAD_IDX].to(DEVICE)
        resid_post = cache["blocks.11.hook_resid_post"].to(DEVICE)

        W_OV = model.W_V[LAYER_IDX, HEAD_IDX].cpu() @ model.W_O[LAYER_IDX, HEAD_IDX].cpu()
        submatrix_of_full_OV_matrix: Float[Tensor, "batch seqK d_vocab"] = W_EE_toks.cpu() @ W_OV.cpu() @ model.W_U.cpu()
        
        # Include self-embedding
        E_sq: Int[Tensor, "batch seqK K_semantic"] = submatrix_of_full_OV_matrix.cpu().topk(5, dim=-1, largest=False).indices 
        E_sq_contains_self: Bool[Tensor, "batch seqK"] = (E_sq == mybatch[:, :, None]).any(-1)
        E_sq[..., -1] = t.where(E_sq_contains_self, E_sq[..., -1], mybatch)

        gc.collect()
        t.cuda.empty_cache()

        loss = get_metric_from_end_state(
            end_state= resid_post,
            model=model,
            targets=mytargets.to(DEVICE),
        )
        
        mean_head_output = head_output.mean(dim=(0, 1))

        mean_ablated_loss = get_metric_from_end_state(
            end_state = resid_post - head_output + mean_head_output[None, None],
            model=model,
            targets=mytargets.to(DEVICE),
        )

        loss_sorted = sorted(
            [
                (batch_idx, seq_idx, (mean_ablated_loss-loss)[batch_idx, seq_idx].item())
                for batch_idx in range(BATCH_SIZE)
                for seq_idx in range(max_seq_len)
            ], 
            key=lambda x: x[2],
            reverse=True,
        )

        loss_to_use = set()
        for i in range((BATCH_SIZE*max_seq_len) // 20):
            loss_to_use.add((loss_sorted[i][0], loss_sorted[i][1]))

        normalized_query = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"].to(DEVICE)
        normalized_query /= (normalized_query.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)

        xs = []
        ys = []
        attn_scores = []
        denoms = []
        biases = []

        warnings.warn("Chaos things")

        for batch_idx in tqdm(range(BATCH_SIZE)):
            for seq_idx in range(1, 200): # max_seq_len): # skip BOS

                if (batch_idx, seq_idx) not in loss_to_use:
                    continue

                # if seq_idx % 20 != 0: # Maybe random sample stops Simpson's paradox?
                #     continue

                denom = torch.exp(attn_score[batch_idx, seq_idx, :seq_idx+1]).sum().log()

                outputs = dot_with_query(
                    unnormalized_keys = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][batch_idx, 1:seq_idx+1].to(DEVICE),
                    unnormalized_queries = einops.repeat(normalized_query[batch_idx, seq_idx], "d -> s d", s=seq_idx),
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx=HEAD_IDX,
                    add_key_bias = True, 
                    add_query_bias = True,
                    normalize_keys = True,
                    normalize_queries = False,
                    use_tqdm=False,
                )

                query = normalized_query[batch_idx, seq_idx]
                wut_indices = E_sq[batch_idx, 1:seq_idx+1].to(DEVICE).transpose(0, 1)

                # print("Sentence looks like", model.to_string(mybatch[batch_idx, :seq_idx+1].cpu().tolist()), "with indices", [(model.to_string(mybatch[batch_idx, 1+s]), model.to_str_tokens(wut_indices[:, s])) for s in range(seq_idx)][:10])
                # Seemed reasonable

                base_parallel, base_perp = project(
                    query, # einops.repeat(query, "d -> s d", s=seq_idx),
                    list(model.W_U.T[logit_lens_topk[batch_idx, seq_idx]]),
                    # list(model.W_U.T[wut_indices]), # Project onto semantically similar tokens
                )

                parallel = einops.repeat(base_parallel, "d -> s d", s=seq_idx)
                perp = einops.repeat(base_perp, "d -> s d", s=seq_idx)

                para_score = dot_with_query(
                    unnormalized_keys = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][batch_idx, 1:seq_idx+1].to(DEVICE),
                    unnormalized_queries = parallel,
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx=HEAD_IDX,
                    add_key_bias = True, 
                    add_query_bias = False,
                    normalize_keys = True,
                    normalize_queries = False,
                    use_tqdm=False,
                )
                perp_score = dot_with_query(
                    unnormalized_keys = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][batch_idx, 1:seq_idx+1].to(DEVICE),
                    unnormalized_queries = perp,
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx=HEAD_IDX,
                    add_key_bias = True, 
                    add_query_bias = False,
                    normalize_keys = True,
                    normalize_queries = False,
                    use_tqdm=False,
                )
                bias_score = dot_with_query(
                    unnormalized_keys = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][batch_idx, 1:seq_idx+1].to(DEVICE),
                    unnormalized_queries = 0.0*perp,
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx=HEAD_IDX,
                    add_key_bias = True, 
                    add_query_bias = True,
                    normalize_keys = True,
                    normalize_queries = False,
                    use_tqdm=False,
                )
                t.testing.assert_close(outputs, para_score + perp_score + bias_score, atol=1e-3, rtol=1e-3)

                indices = outputs.argsort(descending=True)[:5].tolist()
                indices.extend(torch.randperm(seq_idx)[:5].tolist())

                xs.extend((para_score-denom.item())[indices].tolist())
                # ys.append(perp_score.item() - denom.item())
                ys.extend((perp_score-denom.item())[indices].tolist())
                biases.extend((bias_score-denom.item())[indices].tolist())
                denoms.extend([denom.item() for _ in range(len(indices))])
                # attn_scores.append(outputs.item())
                attn_scores.extend((outputs-denom.item())[indices].tolist())
        
        fig = hist(
            [xs, ys],
            # labels={"variable": "Version", "value": "Attn diff (positive ⇒ more attn paid to IO than S1)"},
            title=f"Attention scores for {LAYER_IDX}.{HEAD_IDX}",
            names=["Log attention score from unembedding parallel projection", "Log attention score from unembedding perpendicular projection"],
            width=800,
            height=600,
            opacity=0.7,
            marginal="box",
            template="simple_white",
            return_fig=True,
        )

        fig.show()

# %%
