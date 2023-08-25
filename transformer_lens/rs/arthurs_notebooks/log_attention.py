# %% [markdown] [4]:

"""
Copy of direct effect survey
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import get_metric_from_end_state, dot_with_query
from transformer_lens.rs.callum.explore_prompts.model_results_3 import get_effective_embedding
from transformer_lens.rs.callum2.generate_st_html.utils import (
    ST_HTML_PATH,
)

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
MODE = "Query"

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

if MODE == "Key": # What you want to do here is set `my_embeddings` equal to some residual stream state

    VERSION = "layer 10 context-free residual state"
    # VERSION = "effective embedding"
    my_random_tokens = model.to_tokens("The")[0]
    my_embeddings = t.zeros(BATCH_SIZE, max_seq_len, model.cfg.d_model)

    if VERSION == "effective embedding":
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

    elif VERSION.startswith("layer 10 context-free residual state"):

        mask = torch.eye(max_seq_len).cuda()
        mask[:, 0] += 1
        mask[0, 0] -= 1
        
        score_mask = - mask + 1.0
        assert (score_mask.min().item()) >= 0
        score_mask *= -1000

        gc.collect()
        torch.cuda.empty_cache()

        model.reset_hooks()

        if VERSION == "layer 10 context-free residual state with no positional embeddings":
            model.add_hook(
                "hook_pos_embed",
                lambda z, hook: 0.0*z,
            )

        for layer_idx in range(LAYER_IDX):
            
            # model.add_hook(
            #     f"blocks.{layer_idx}.attn.hook_attn_scores",
            #     lambda z, hook: z + score_mask[None, None], # kill all but BOS and current token
            # )

            # model.add_hook( 
            #     f"blocks.{layer_idx}.attn.hook_pattern",
            #     lambda z, hook: (z * mask[None, None].cuda()) / (0.00001 + 0.5*mask[None, None].cpu()*cache[f"blocks.{hook.layer()}.attn.hook_pattern"]).mean(dim=0, keepdim=True).cuda(), # scale so that the total attention paid is the average attention paid across the batch (20); could also try batch and seq...
            # )

            model.add_hook( # This is the only thing that works; other rescalings suggest that the perpendicular component is more important. It also seems the other interventions just totally broke?
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

fpath = ST_HTML_PATH.parent.parent / "cspa/cspa_semantic_dict_full_token_idx_version.pkl"
token_idx_version = torch.load(fpath)

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
    break

#%%

att_values = []

if True: # remove at some point in future 
    for batch_idx in tqdm(range(BATCH_SIZE)):
        for seq_idx in range(1, 200): # max_seq_len): # skip BOS

            if (batch_idx, seq_idx) not in loss_to_use:
                continue

            old_denom_items = attn_score[batch_idx, seq_idx, :seq_idx+1]

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
            
            ADD_KEY_BIAS = True

            if MODE == "Key":
                para_keys, perp_keys = project(
                    normalized_keys,
                    my_embeddings[batch_idx, 1:seq_idx+1].to(DEVICE),
                )

                para_score = dot_with_query(
                    unnormalized_keys = para_keys,
                    unnormalized_queries = einops.repeat(normalized_query[batch_idx, seq_idx], "d -> s d", s=seq_idx),
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx = HEAD_IDX,
                    add_key_bias = ADD_KEY_BIAS,
                    add_query_bias=True,
                    normalize_keys = False,
                    normalize_queries = True,
                )
                perp_score = dot_with_query(
                    unnormalized_keys = perp_keys,
                    unnormalized_queries = einops.repeat(normalized_query[batch_idx, seq_idx], "d -> s d", s=seq_idx),
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx = HEAD_IDX,
                    add_key_bias = ADD_KEY_BIAS,
                    add_query_bias=True,
                    normalize_keys = False,
                    normalize_queries = True,
                )
                bias_score = dot_with_query(
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

            elif MODE == "Query":
                VERSION = "single_unembedding"
                if VERSION == "old_semantically_similar":
                    
                    raise NotImplementedError("This is broken, need to fix the off by 1 we get cos avoiding BOS")

                    K_semantic = 10
                    W_QK = model.W_Q[LAYER_IDX, HEAD_IDX].cpu() @ model.W_K[LAYER_IDX, HEAD_IDX].T.cpu() / (model.cfg.d_head ** 0.5)
                    
                    current_ee = W_EE_toks[batch_idx, :seq_idx+1].cpu()
                    normalized_current_ee = current_ee / (current_ee.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)

                    times_by_qk = normalized_current_ee @ W_QK.T.cpu()
                    times_by_wu = times_by_qk @ model.W_U.cpu()

                    # submatrix_of_full_QK_matrix: Float[Tensor, "batch seqK d_vocab"] = W_EE_toks.cpu() @ W_QK.T.cpu() @ model.W_U.cpu()


                    E_sq_QK: Int[Tensor, "seqK K_semantic"] = times_by_wu.topk(K_semantic, dim=-1).indices

                    E_sq_QK_contains_self: Bool[Tensor, "seqK"] = (E_sq_QK == mybatch[batch_idx, :1+seq_idx, None]).any(-1)
                    E_sq_QK[..., -1] = t.where(E_sq_QK_contains_self, E_sq_QK[..., -1], mybatch[batch_idx, :1+seq_idx])
                    E_sq_QK_rearranged = einops.rearrange(E_sq_QK, "seqK K_semantic -> K_semantic seqK") # K_semantic first is easier for projection

                elif VERSION == "new_hardcoded_semantically_similar":
                    # K_semantic = 8 # ugh just include all, there are a non constant n7umber...
                    relevant_tokens = mybatch[batch_idx, 1:1+seq_idx]
                    try:
                        sem_sim_things = [
                            torch.LongTensor(list(token_idx_version[token.item()])).to(normalized_query.device) for token in relevant_tokens # ughhh these are 
                        ]
                        print("Success")
                    except Exception as e:
                        bad_pairs.append((batch_idx, seq_idx))
                        print("Failure", e)
                        continue

                else:
                    assert VERSION == "single_unembedding"

                query = normalized_query[batch_idx, seq_idx]

                if VERSION == "new_hardcoded_semantically_similar":
                    base_parallel = torch.zeros((seq_idx, model.cfg.d_model)).to(query.device)
                    base_perp = torch.zeros((seq_idx, model.cfg.d_model)).to(query.device)
                    for project_seq_idx in range(1, seq_idx+1):
                        elem_base_parallel, elem_base_perp = project(
                            query.clone(),
                            list(model.W_U.T[sem_sim_things[project_seq_idx-1]]),
                        )
                        base_parallel[project_seq_idx-1] = elem_base_parallel
                        base_perp[project_seq_idx-1] = elem_base_perp

                else:
                    base_parallel, base_perp = project(
                        einops.repeat(query, "d -> s d", s=seq_idx),
                        model.W_U.T[mybatch[batch_idx, 1:1+seq_idx]] if VERSION != "old_semantically_similar" else list(model.W_U.T[E_sq_QK_rearranged.cuda()]),
                        # list(model.W_U.T[logit_lens_topk[batch_idx, seq_idx]]),
                        # list(model.W_U.T[wut_indices]), # Project onto semantically similar tokens
                    )
                
                parallel = base_parallel
                perp = base_perp

                para_score = dot_with_query(
                    unnormalized_keys = cache[f"blocks.{LAYER_IDX}.hook_resid_pre"][batch_idx, 1:seq_idx+1].to(DEVICE),
                    unnormalized_queries = parallel,
                    model=model,
                    layer_idx=LAYER_IDX,
                    head_idx=HEAD_IDX,
                    add_key_bias = True, 
                    add_query_bias = True,
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
                    add_query_bias = True,
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

                warnings.warn("We reenabled query biases so can't do the sanity check!")
                # t.testing.assert_close(outputs, para_score + perp_score + bias_score, atol=1e-3, rtol=1e-3)

            SZ = 2
            sorted_indices = outputs.argsort(descending=True)[:SZ].tolist()
            indices = torch.tensor(sorted_indices[:SZ]) + 1
            att_values.extend(attn_pattern[batch_idx, seq_idx, indices].tolist())

            # We also should probably recompute the denominator : ) git blame for old way where we did not recompute this
            
            SUBTRACT_BASELINE = True
            RECOMPUTE_DENOM = True

            for score, xs_or_ys in zip(
                [para_score, perp_score],
                [xs, ys],
                strict=True,
            ):
                denom_items = einops.repeat(old_denom_items, "seq_len -> indices_length seq_len", indices_length=len(indices)).clone().cpu()

                if RECOMPUTE_DENOM:
                    denom_items[torch.arange(len(indices)), indices] = score[indices-1]

                new_denom = torch.exp(denom_items).sum(dim=-1, keepdim=False)
                xs_or_ys.extend((torch.exp(score[indices-1]) / new_denom - (0.0 if not SUBTRACT_BASELINE else attn_pattern[batch_idx, seq_idx, indices].cpu())).tolist()) # 1+ accounts for zeroindexing

            used_batch_indices.extend([batch_idx for _ in range(len(indices))])
            used_seq_indices.extend([seq_idx for _ in range(len(indices))])

            if not ADD_KEY_BIAS:
                t.testing.assert_close(outputs, para_score + perp_score + bias_score, atol=1e-3, rtol=1e-3)

            bias_scores.extend(bias_score[indices-1].tolist())
            
            used_batch_indices.extend([batch_idx for _ in range(len(indices))])
            used_seq_indices.extend([seq_idx for _ in range(len(indices))])

    # Make two plots

    r2 = np.corrcoef(xs, ys)[0, 1] ** 2

    # get best fit line 
    m, b = np.polyfit(xs, ys, 1)

    df = pd.DataFrame({
        'xs': xs,
        'ys': ys,
        # #  very sad, broken
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
        title = f"Change in attention probabilities when projecting Head {LAYER_IDX}.{HEAD_IDX} {' ' if MODE == 'Query' and VERSION == 'single_unembedding' else ' (names inaccurate!)'}",
        xaxis_title="Project query to unembedding parallel projection",
        yaxis_title="Project query onto unembedding complement",
    )
    fig.show()

    fig = hist(
        [xs],
        labels={"variable": MODE + " input component", "value": "Change in attention"},
        title=f"Change in {LAYER_IDX}.{HEAD_IDX} attention probabilities when approximating {MODE.lower()} with {VERSION.replace('_', ' ')}",
        # names=["Parallel", "Perpendicular"], # we removed support for both at same time
        width=1200,
        height=600,
        opacity=0.7,
        marginal="box",
        nbins = 20,
        template="simple_white",
        return_fig=True,
    )
    
    # # Sad, kills the box plot
    # # Update y-axis
    # fig.update_yaxes(range=[-0.1, max_count]) # Set the y-axis zoom level

    # Computing the histograms for xs and ys
    counts_xs, _ = np.histogram(xs)
    counts_ys, _ = np.histogram(ys)

    # Getting the maximum count
    max_count = max(max(counts_xs), max(counts_ys))

    # add dotted x = 0 line 
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, 1.5*max_count],
            mode="lines",
            name="x = 0",
            marker=dict(color="black"),
        )
    )

    fig.show()

    old_xs = deepcopy(xs)
    old_ys = deepcopy(ys)
    assert False, "Usually OOMs after this cell is reran, so this is a good place to stop and restart the kernel"

# %%
