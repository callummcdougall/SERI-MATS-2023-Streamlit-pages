# %% [markdown] [4]:

"""
Copy of direct effect survey
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_metric_from_end_state, dot_with_query
import argparse

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

# %%

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
model.to("cuda")

#%%

for LAYER_IDX, HEAD_IDX in [(10, 7)] +  list(itertools.product(range(9, 12), range(12))):     
        attn_score = cache[f"blocks.{LAYER_IDX}.attn.hook_attn_scores"][:, HEAD_IDX].to(DEVICE)
        head_output = cache[f"blocks.{LAYER_IDX}.attn.hook_result"][:, :, HEAD_IDX].to(DEVICE)
        resid_post = cache["blocks.11.hook_resid_post"].to(DEVICE)

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
                parallel, perp = project(
                    einops.repeat(query, "d -> s d", s=seq_idx),
                    model.W_U.T[mybatch[batch_idx, 1:seq_idx+1]],
                )

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

        # get correlation
        r2 = np.corrcoef(xs, ys)[0, 1] ** 2

        # get best fit line 
        m, b = np.polyfit(xs, ys, 1)

        fig = px.scatter(
            x=xs,
            y=ys,
        )

        # add best fit line from min x to max x
        fig.add_trace(
            go.Scatter(
                x=[min(xs), max(xs)],
                y=[m * min(xs) + b, m * max(xs) + b],
                mode="lines",
                name=f"r^2 = {r2:.3f}",
            )
        )

        # add y = mx + c label
        fig.add_annotation(
            x=0.1,
            y=0.1,
            text=f"y = {m:.3f}x + {b:.3f}",
            showarrow=False,
        )

        fig.update_layout(
            title = f"Comparison of attention scores from parallel and perpendicular projections for Head {LAYER_IDX}.{HEAD_IDX}",
            xaxis_title="Log attention score from unembedding parallel projection",
            yaxis_title="Log attention score from unembedding perpendicular projection",
        )

        fig.show()

# %%

