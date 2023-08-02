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

LAYER_IDX = 10
HEAD_IDX = 7

model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)
model.set_use_attn_in(True)
DEVICE = "cuda"
SHOW_PLOT = True
DATASET_SIZE = 500
BATCH_SIZE = 30 
MODE = "key"

#%%

dataset = get_webtext(seed=6)
max_seq_len = model.tokenizer.model_max_length

# %%

filtered_tokens = []
targets = []  # targets for prediction

new_batch_to_old_batch = {}

print("Not rapid, but not THAT slow :-) ")
_idx = -1
while len(filtered_tokens) < DATASET_SIZE:
    _idx += 1
    cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
    if (
        len(cur_tokens) > max_seq_len
    ):  # so we're not biasing towards early sequence positions...
        new_batch_to_old_batch[len(filtered_tokens)] = _idx
        filtered_tokens.append(cur_tokens[:max_seq_len])
        targets.append(cur_tokens[1 : max_seq_len + 1])

mybatch = torch.LongTensor(filtered_tokens[:BATCH_SIZE])
mytargets = torch.LongTensor(targets[:BATCH_SIZE])

#%%

logits, cache = model.run_with_cache(
    mybatch.to(DEVICE), 
    # names_filter = lambda name: name in [f"blocks.{LAYER_IDX}.hook_resid_pre", f"blocks.{LAYER_IDX}.attn.hook_attn_scores", f"blocks.{LAYER_IDX}.attn.hook_result", "blocks.11.hook_resid_post"],
    names_filter = lambda name: name.endswith(("hook_attn_scores", "hook_pattern", "hook_result", "hook_resid_pre", ".11.hook_resid_post", "blocks.0.hook_resid_post")),
    device="cpu",
)

#%%

resid_pre = cache["blocks.10.hook_resid_pre"]
scaled_resid_pre = resid_pre / (resid_pre.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)

logit_lens = einops.einsum(
    scaled_resid_pre.to(DEVICE),
    model.W_U,
    "batch seq_len d_model, d_model d_vocab -> batch seq_len d_vocab",
).cpu()

#%%

logit_lens_topk = logit_lens.topk(5, dim=-1).indices.cuda()

#%%

W_EE = get_effective_embedding(model)['W_E (including MLPs)']

#%%

if MODE == "key":
    VERSION = "ee"
    my_random_tokens = model.to_tokens("The")[0]
    my_embeddings = t.zeros(BATCH_SIZE, max_seq_len, model.cfg.d_model)

    if VERSION == "ee":
        my_embeddings[:] = W_EE.cpu()[mybatch]

    elif VERSION == "the":
        print("Making the embeddings...")
        for batch_idx in tqdm(range(BATCH_SIZE)):
            current_prompt = t.cat([
                einops.repeat(my_random_tokens, "random_seq_len -> cur_seq_len random_seq_len", cur_seq_len=max_seq_len).clone().cpu(),
                mybatch[batch_idx].unsqueeze(-1).clone().cpu(),
            ],dim=1)
            
            gc.collect()
            t.cuda.empty_cache()

            current_embeddings = model.run_with_cache(
                current_prompt.to(DEVICE),
                names_filter = lambda name: name==get_act_name("resid_pre", 10),
            )[1][get_act_name("resid_pre", 10)][torch.arange(max_seq_len), -1].cpu()
            my_embeddings[batch_idx] = current_embeddings

    elif VERSION == "mlp0":
        my_embeddings[:] = cache[get_act_name("resid_post", 0)].cpu()

    elif VERSION == "callum":

        mask = torch.eye(max_seq_len).cuda()
        mask[:, 0] += 1
        mask[0, 0] -= 1
        
        score_mask = - mask + 1.0
        assert (score_mask.min().item()) >= 0
        score_mask *= -1000

        gc.collect()
        torch.cuda.empty_cache()

        model.reset_hooks()
        for layer_idx in range(LAYER_IDX):
            
            # model.add_hook(
            #     f"blocks.{layer_idx}.attn.hook_attn_scores",
            #     lambda z, hook: z + score_mask[None, None], # kill all but BOS and current token
            # )

            # model.add_hook( 
            #     f"blocks.{layer_idx}.attn.hook_pattern",
            #     lambda z, hook: (z * mask[None, None].cuda()) / (0.5 * mask[None, None].cpu()*cache[f"blocks.{hook.layer()}.attn.hook_pattern"]).sum(dim=-1, keepdim=True).mean(dim=0, keepdim=True).cuda(), # scale so that the total attention paid is the average attention paid across the batch (20); could also try batch and seq...
            #     level=1,
            # )

            model.add_hook( # # This is the only thing that works; other rescalings suggest that the perpendicular component is more important
                f"blocks.{layer_idx}.attn.hook_pattern",
                lambda z, hook: (z * mask[None, None].cuda()),
            )

        cached_hook_resid_pre = model.run_with_cache(
            mybatch.to(DEVICE),
            names_filter = lambda name: name==get_act_name("resid_pre", 10),
        )[1][get_act_name("resid_pre", 10)].cpu()
        my_embeddings[:] = cached_hook_resid_pre.cpu()
        del cached_hook_resid_pre
        gc.collect()
        t.cuda.empty_cache()

    else: 
        raise Exception("Invalid version")

# %%

W_EE_toks: Float[Tensor, "batch seqK d_model"] = W_EE[mybatch]

#%%

for LAYER_IDX, HEAD_IDX in [(10, 7)] +  list(itertools.product(range(9, 12), range(12))):
    attn_score = cache[f"blocks.{LAYER_IDX}.attn.hook_attn_scores"][:, HEAD_IDX].to(DEVICE)
    attn_pattern = attn_score.exp() / attn_score.exp().sum(dim=-1, keepdim=True)
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

    unnormalized_query = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"].to(DEVICE)
    normalized_query = unnormalized_query / (unnormalized_query.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)

    xs = []
    ys = []

    used_batch_indices = []
    used_seq_indices = []
    bias_scores = []

    for batch_idx in tqdm(range(BATCH_SIZE)):
        for seq_idx in range(1, 200): # max_seq_len): # skip BOS

            if (batch_idx, seq_idx) not in loss_to_use:
                continue

            denom = torch.exp(attn_score[batch_idx, seq_idx, :seq_idx+1]).sum()

            unnormalized_keys = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][batch_idx, 1:seq_idx+1].to(DEVICE)
            normalized_keys = unnormalized_keys / (unnormalized_keys.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5) 

            outputs = dot_with_query(
                unnormalized_keys = normalized_keys,
                unnormalized_queries = einops.repeat(normalized_query[batch_idx, seq_idx], "d -> s d", s=seq_idx),
                model=model,
                layer_idx=LAYER_IDX,
                head_idx=HEAD_IDX,
                add_key_bias = True, 
                add_query_bias = True,
                normalize_keys = False,
                normalize_queries = False,
                use_tqdm=False,
            )
            
            if MODE == "key":
                para_keys, perp_keys = project(
                    normalized_keys,
                    my_embeddings[batch_idx, 1:seq_idx+1].to(DEVICE),
                )

                the_para_score = dot_with_query(
                    unnormalized_keys = para_keys,
                    unnormalized_queries = einops.repeat(normalized_query[batch_idx, seq_idx], "d -> s d", s=seq_idx),
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx = HEAD_IDX,
                    add_key_bias = False,
                    add_query_bias=True,
                    normalize_keys = False,
                    normalize_queries = True,
                )
                the_perp_score = dot_with_query(
                    unnormalized_keys = perp_keys,
                    unnormalized_queries = einops.repeat(normalized_query[batch_idx, seq_idx], "d -> s d", s=seq_idx),
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx = HEAD_IDX,
                    add_key_bias = False,
                    add_query_bias=True,
                    normalize_keys = False,
                    normalize_queries = True,
                )
                the_key_bias_score = dot_with_query(
                    unnormalized_keys = 0.0*perp_keys,
                    unnormalized_queries = einops.repeat(normalized_query[batch_idx, seq_idx], "d -> s d", s=seq_idx),
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx = HEAD_IDX,
                    add_key_bias = True,
                    add_query_bias=True,
                    normalize_keys = False,
                    normalize_queries = True,
                )

                t.testing.assert_close(outputs, the_para_score + the_perp_score + the_key_bias_score, atol=1e-3, rtol=1e-3)

                indices = outputs.argsort(descending=True)[:3].tolist()
                # indices.extend(torch.randperm(seq_idx)[:3].tolist())

                xs.extend((torch.exp(the_para_score) / denom.item()))
                ys.extend((torch.exp(the_perp_score) / denom.item())) #   - attn_pattern[batch_idx, seq_idx, 0].item() )[indices].tolist())
                bias_scores.extend((torch.exp(the_key_bias_score) / denom.item())[indices].tolist())

                used_batch_indices.extend([batch_idx for _ in range(len(indices))])
                used_seq_indices.extend([seq_idx for _ in range(len(indices))])

            elif MODE == "query":
                K_semantic = 10
                W_QK = model.W_Q[LAYER_IDX, HEAD_IDX].cpu() @ model.W_K[LAYER_IDX, HEAD_IDX].T.cpu() / (model.cfg.d_head ** 0.5)
                
                current_ee = W_EE_toks[batch_idx, 1:seq_idx+1].cpu()
                normalized_current_ee = current_ee / (current_ee.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)

                times_by_qk = normalized_current_ee @ W_QK.T.cpu()
                times_by_wu = times_by_qk @ model.W_U.cpu()

                # submatrix_of_full_QK_matrix: Float[Tensor, "batch seqK d_vocab"] = W_EE_toks.cpu() @ W_QK.T.cpu() @ model.W_U.cpu()

                E_sq_QK: Int[Tensor, "seqK K_semantic"] = times_by_wu.topk(K_semantic, dim=-1).indices

                E_sq_QK_contains_self: Bool[Tensor, "seqK"] = (E_sq_QK == mybatch[batch_idx, 1:1+seq_idx, None]).any(-1)
                E_sq_QK[..., -1] = t.where(E_sq_QK_contains_self, E_sq_QK[..., -1], mybatch[batch_idx, 1:1+seq_idx])
                E_sq_QK_rearranged = einops.rearrange(E_sq_QK, "seqK K_semantic -> K_semantic seqK") # K_semantic first is easier for projection

                query = normalized_query[batch_idx, seq_idx]

                base_parallel, base_perp = project(
                    einops.repeat(query, "d -> s d", s=seq_idx),
                    list(model.W_U.T[E_sq_QK_rearranged.cuda()]),
                    # list(model.W_U.T[logit_lens_topk[batch_idx, seq_idx]]),
                    # list(model.W_U.T[wut_indices]), # Project onto semantically similar tokens
                    # model.W_U.T[mybatch[batch_idx, 1:1+seq_idx]],
                )
                # parallel = einops.repeat(base_parallel, "d -> s d", s=seq_idx)
                # perp = einops.repeat(base_perp, "d -> s d", s=seq_idx)

                parallel = base_parallel
                perp = base_perp

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

                indices = outputs.argsort(descending=True)[:2].tolist()
                # indices.extend(torch.randperm(seq_idx)[:5].tolist())

                xs.extend((torch.exp(para_score) / denom.item())[indices].tolist())
                # xs.extend((para_score - denom.log().item())[indices].tolist())

                ys.extend((torch.exp(perp_score) / denom.item())[indices].tolist())
                # ys.extend((perp_score - denom.log().item())[indices].tolist())

                used_batch_indices.extend([batch_idx for _ in range(len(indices))])
                used_seq_indices.extend([seq_idx for _ in range(len(indices))])

                # bias?
                
                # if max(ys[-len(perp_score):]) > 5.0: # This was a set of extreme values at some point
                #     print("Max index", ys[-len(perp_score):].index(max(ys[-len(perp_score):])))
                #     print("Sentence looks like", model.to_string(mybatch[batch_idx, :seq_idx+1].cpu().tolist()), "with indices",  model.to_str_tokens(logit_lens_topk[batch_idx, seq_idx]), [(s, perp_score[s-1].item(), model.to_string(mybatch[batch_idx, s])) for s in range(1, seq_idx+1)])
                #     # Seemed reasonable # 

                bias_scores.extend((torch.exp(bias_score) / denom.item()).tolist())
                # (bias_score-denom.item())[indices].tolist())
                # denoms.extend([denom.item() for _ in range(len(indices))])
                # attn_scores.append(outputs.item())
            

    if True: # These correlation plots do not work
        r2 = np.corrcoef(xs, ys)[0, 1] ** 2

        # get best fit line 
        m, b = np.polyfit(xs, ys, 1)

        df = pd.DataFrame({
        'xs': xs,
        'ys': ys,
        # sad, below is so broken
        # 'text': [(str(model.to_string(mybatch[used_batch_idx, :used_seq_idx+1].cpu().tolist()))[-20:], 
        #          "with completion", 
        #          model.to_string(mytargets[used_batch_idx, used_seq_idx:used_seq_idx+1]), new_batch_to_old_batch[used_batch_idx], used_seq_idx) for used_batch_idx, used_seq_idx in zip(used_batch_indices, used_seq_indices, strict=True)]
        })

        fig = px.scatter(
            df,
            x='xs',
            y='ys',
            # hover_data=['text'] # sad broken
        )

        # add best fit line from min x to max x
        fig.add_trace(
            go.Scatter(
                x=[min(xs), max(xs)],
                y=[m * min(xs) + b, m * max(xs) + b],
                # text = 
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

    else:        
        fig = hist(
            [xs, ys],
            # labels={"variable": "Version", "value": "Attn diff (positive ⇒ more attn paid to IO than S1)"},
            title=f"Attention scores for {LAYER_IDX}.{HEAD_IDX}",
            names=["Attention probability contribution from unembedding parallel projection", "Attention probability contribution from unembedding perpendicular projection"] if MODE == "query" else ["Para", "perp"],
            width=800,
            height=600,
            opacity=0.7,
            marginal="box",
            template="simple_white",
            return_fig=True,
        )
        fig.show()

    old_xs = deepcopy(xs)
    old_ys = deepcopy(ys)

# %%
