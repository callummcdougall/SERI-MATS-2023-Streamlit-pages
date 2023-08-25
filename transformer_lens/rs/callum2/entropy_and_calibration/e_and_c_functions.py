from transformer_lens.cautils.utils import *

from transformer_lens.rs.callum2.generate_st_html.model_results import get_result_mean
from transformer_lens.rs.callum2.utils import update_mean

# from tqdm.notebook import tqdm_notebook


# ====================================================================================================
# ! Entropy functions
# ====================================================================================================

def entropy(logits: Float[Tensor, "... d_vocab"]):
    logprobs = logits.log_softmax(dim=-1)
    return -(logprobs.exp() * logprobs).sum(dim=-1)


def entropy_measure(
    model: HookedTransformer,
    toks: Int[Tensor, "batch_size seq_len"],
    minibatch_size: Optional[int] = None,
    include_mlps: bool = False,
    result_mean: Optional[dict] = None,
):
    '''
    Measures direct entropy contribution from every component in the model, both relative to the final logits, and relative to the
    current value in the residual stream (i.e. the logit lens).

    Optionally can also measure the contribution of MLPs, but defaults to just attention heads.    
    '''
    # Split tokens, to avoid memory errors
    toks_split: List[Tensor] = toks.split(minibatch_size) if minibatch_size is not None else [toks]

    # Get mean results
    result_mean = get_result_mean(
        head_list = None,
        toks = toks,
        model = model,
        minibatch_size = minibatch_size,
        keep_seq_dim = True,
        include_mlps = include_mlps,
        verbose = True,
    ) if result_mean is None else result_mean

    # Define useful variables
    device = toks.device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    n_components_per_layer = n_heads + (1 if include_mlps else 0)
    n_resid_states = n_layers * (2 if include_mlps else 1) + 1
    components_to_cache = ["resid_pre", "result", "scale"] + (["mlp_out", "resid_mid"] if include_mlps else [])
    W_U = model.W_U
    
    # Get tensor to store the results
    entropy_diffs = t.zeros(n_layers, n_components_per_layer).to(device) # = difference in logit-lens entropy
    entropy_marginals = t.zeros(n_layers, n_components_per_layer).to(device) # = marginal effect on final entropy
    resid_entropies = t.zeros(n_resid_states).to(device)

    progress_bar = tqdm(total = n_layers * len(toks_split))
    num_toks_seen = 0

    for _toks in toks_split:
        num_toks = _toks.numel()

        # Run with cache, getting all the residual stream & output components we need to calc entropy
        logits, cache = model.run_with_cache(
            _toks,
            names_filter = lambda name: any([name.endswith(x) for x in components_to_cache])
        )
        # Get final entropy, and fill it in (using the standard update rule for the mean)
        logits: Tensor = logits.unsqueeze(-2) # shape [batch seq 1 d_vocab], so it can broadcast along component dimension
        entropy_final = entropy(logits)
        resid_entropies[-1] = update_mean(resid_entropies[-1], entropy_final.mean(), num_toks_seen, num_toks)
        
        # ! For each layer, we calculate the logit lens (before attention heads and before MLPs), we also get entropy
        for layer in range(n_layers):

            # If include_mlps, then we need to measure the entropy of resid_mid as well as resid_pre
            if include_mlps:
                resid_pre_logits = (cache["resid_pre", layer] / cache["scale"]).unsqueeze(-2) @ W_U # [batch seq 1 d_vocab]
                resid_mid_logits = (cache["resid_mid", layer] / cache["scale"]).unsqueeze(-2) @ W_U # [batch seq 1 d_vocab]
                resid_pre_entropy = entropy(resid_pre_logits)
                resid_mid_entropy = entropy(resid_mid_logits)
                resid_entropies[2 * layer] = update_mean(resid_entropies[2 * layer], resid_pre_entropy.mean(), num_toks_seen, num_toks)
                resid_entropies[2 * layer + 1] = update_mean(resid_entropies[2 * layer + 1], resid_mid_entropy.mean(), num_toks_seen, num_toks)
            else:
                resid_pre_logits = (cache["resid_pre", layer] / cache["scale"]).unsqueeze(-2) @ W_U # [batch seq 1 d_vocab]
                resid_pre_entropy = entropy(resid_pre_logits)
                resid_entropies[layer] = update_mean(resid_entropies[layer], resid_pre_entropy.mean(), num_toks_seen, num_toks)

            # ! For each head, we calculate the marginal entropy from ablating it (we do this all at once). Also append MLPs if necessary.

            # Get all component contributions
            components_mean_contribution = t.stack([result_mean[(layer, head)] for head in range(n_heads)], dim=1) # [seq 1 d_model]
            component_resid_contribution = cache["result", layer] - components_mean_contribution # [batch seq head d_model]
            if include_mlps:
                mlp_mean_contribution = result_mean[layer]
                mlp_contribution = cache["mlp_out", layer] - mlp_mean_contribution # [batch seq d_model]
                component_resid_contribution = t.stack([component_resid_contribution, mlp_contribution.unsqueeze(-2)], dim=-2) # [batch seq head+1 d_model]

            # Get logit lens contributions
            component_logits = (component_resid_contribution / cache["scale"].unsqueeze(-1)) @ W_U # [batch seq head+1 d_vocab]

            # Calculate the entropy diff from this layer's components, and update mean using the standard update rule for mean
            new_entropy = entropy(resid_pre_logits + component_logits) # [batch seq head+1]
            entropy_diff = new_entropy - resid_pre_entropy
            entropy_diff_mean = entropy_diff.mean(dim=0).mean(dim=0) # [head+1]
            entropy_diffs[layer] = update_mean(entropy_diffs[layer], entropy_diff_mean, num_toks_seen, num_toks)

            # Calculate the marginal entropy from this layer's components, and update mean using the standard update rule for mean
            ablated_entropy = entropy(logits - component_logits)
            entropy_marginal = entropy_final - ablated_entropy
            entropy_marginal_mean = entropy_marginal.mean(dim=0).mean(dim=0) # [head+1]
            entropy_marginals[layer] = update_mean(entropy_marginals[layer], entropy_marginal_mean, num_toks_seen, num_toks)

            progress_bar.update()
            gc.collect(); t.cuda.empty_cache()
        
        num_toks_seen += num_toks

    return resid_entropies, entropy_diffs, entropy_marginals



def concat_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]



def make_entropy_resid_plots(
    resid_entropies: Float[Tensor, "resid_position"],
    static: bool = False,
):
    resid_entropies_diffs = resid_entropies[1:] - resid_entropies[:-1]
    resid_entropies_attn_diffs, resid_entropies_mlp_diffs = resid_entropies_diffs[::2].tolist(), resid_entropies_diffs[1::2].tolist()

    # labels = concat_lists([[f"Attn {i}", f"MLP {i}"] for i in range(model.cfg.n_layers)])
    # print(labels)
    line(
        [resid_entropies_attn_diffs, resid_entropies_mlp_diffs], 
        width=600, 
        title="Increase in entropy at each layer & component (logit lens)", 
        labels={"value": "Entropy diff", "index": "Layer", "variable": "Component"},
        template="simple_white",
        names=["Attention", "MLPs"],
        static=static
    )



def make_entropy_plots(
    entropy_diffs: Float[Tensor, "layers heads_and_mlps"],
    entropy_marginals: Float[Tensor, "layers heads_and_mlps"],
    model: HookedTransformer,
    title: Optional[str] = None,
):
    (layers, heads_and_mlps) = entropy_marginals.shape

    assert layers == model.cfg.n_layers
    assert heads_and_mlps == model.cfg.n_heads + 1
    assert entropy_diffs.shape == entropy_marginals.shape

    entropy = t.stack([entropy_diffs, entropy_marginals], dim=0)

    title = f" ({title})" if (title is not None) else ""
    fig_list = []
    fig_list.append(imshow(
        entropy, 
        facet_col=0,
        facet_labels=["Diff (pre/post)", "Marginal (wrt final logits)"],
        width=1000,
        title="Reduction in entropy as a consequence of each head" + title,
        border=True,
        labels={"x": "Heads (+ MLP)", "y": "Layer"},
        draw=True,
        return_fig=True
    ))

    zmax = entropy[..., :-1].abs().max().item()
    fig_list.append(imshow(
        entropy[..., :-1],
        facet_col=0,
        facet_labels=["Diff (pre/post)", "Marginal (wrt final logits)"],
        width=1000, 
        title="Remove MLPs" + title, 
        border=True, 
        zmin=-zmax, 
        zmax=zmax, 
        labels={"x": "Heads", "y": "Layer"},
        draw=True,
        return_fig=True
    ))

    entropy_increases = entropy[..., :-1] * (entropy[..., :-1] > 0)
    zmax = entropy_increases.max().item()
    fig_list.append(imshow(
        entropy_increases, 
        facet_col=0,
        facet_labels=["Diff (pre/post)", "Marginal (wrt final logits)"],
        width=1000, 
        title="Only showing entropy increases" + title, 
        border=True, 
        zmin=-zmax, 
        zmax=zmax, 
        labels={"x": "Heads", "y": "Layer"},
        draw=True,
        return_fig=True
    ))

    return fig_list







# ====================================================================================================
# ! Calibration functions
# ====================================================================================================

def perfect_calibration_line(n_bins) -> Float[Tensor, "n_bins"]:
    return (t.arange(n_bins) + 0.5) / n_bins

def perfect_overconfidence_line(n_bins) -> Float[Tensor, "n_bins"]:
    return t.full((n_bins,), fill_value=0.5)

def my_line(
    y: Float[Tensor, "n_bins"],
    y_upper: Optional[Float[Tensor, "n_bins"]] = None,
    y_lower: Optional[Float[Tensor, "n_bins"]] = None,
    title: Optional[str] = None,
):
    x = list(range(len(y)))
    fig = go.Figure([go.Scatter(
        y=y,
        x=x,
        line=dict(color='rgb(0,100,80)'),
        mode='lines'
    )])
    if (y_upper is not None) and (y_lower is not None):
        fig.add_trace(go.Scatter(
            x=x+x[::-1], # x, then x reversed
            y=y_upper+y_lower[::-1], # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
    if min(y) >= 0:
        yaxis_range = [0, 1]
    else:
        y_max = max(max(y_upper), -min(y_lower))
        yaxis_range = [-y_max, y_max]
    fig.update_layout(
        width=800,
        height=600,
        showlegend=False,
        title=title if title is not None else "",
        xaxis_range=[0, len(y)-1],
        yaxis_range=yaxis_range
    )
    fig.show()


def calculate_overconfidence(accuracy: Float[Tensor, "n_bins"]):
    '''
    Quantifies overconfidence from an accuracy curve.

    This metric is:
        0 if the model is perfectly calibrated (i.e. says something has X% prob implies it always happens with frequency X)
        1 if the model is as overconfident as it could be without being inaccurate (it always assigns 50% probability to the less likely outcome)
        -1 if the model is as underconfident as it could be without being inaccurate
    
    underconfident_sigmoid = t.linspace(-5, 5, 100).sigmoid()
    calculate_overconfidence(underconfident_sigmoid) -> about -0.7
    '''
    n_bins = len(accuracy)

    f = accuracy - perfect_calibration_line(n_bins).to(accuracy.device)
    x = perfect_overconfidence_line(n_bins).to(accuracy.device) - perfect_calibration_line(n_bins).to(accuracy.device)

    return (f * x).sum() / (x * x).sum()


def measure_calibration(
    toks: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    minibatch_size: Optional[int] = None,
    top_k: int = 3,
    n_bins: int = 50,
    plot_calibration_curve: bool = False,
    return_buckets: bool = False,
):
    bucket_frequency = t.zeros(n_bins).to(toks.device)
    bucket_correct_frequency = t.zeros(n_bins).to(toks.device)

    toks_split: List[Tensor] = toks.split(minibatch_size) if minibatch_size is not None else [toks]

    for _toks in tqdm(toks_split):

        gc.collect(); t.cuda.empty_cache()

        logits: Float[Tensor, "batch seq d_vocab"] = model(_toks, return_type="logits")[:, :-1]
        probs = logits.softmax(-1)

        next_toks = _toks[:, 1:]

        topk_probs = probs.topk(k=top_k, dim=-1)
        topk_probs_values: Float[Tensor, "batch seq k"] = topk_probs.values
        topk_probs_indices: Int[Tensor, "batch seq k"] = topk_probs.indices
        
        # Figure out all the indices where the model was correct
        mask_is_correct: Bool[Tensor, "batch seq k"] = (topk_probs_indices == next_toks.unsqueeze(-1))

        for i in t.arange(n_bins):
            p_lower = i / n_bins
            p_upper = (i + 1) / n_bins
            # Figure out the indices of all values which are predicted for this probability
            mask_is_in_probability_bucket = (topk_probs_values >= p_lower) & (topk_probs_values < p_upper)
            # Calculate how many were correct
            bucket_frequency[i] += mask_is_in_probability_bucket.sum()
            bucket_correct_frequency[i] += (mask_is_in_probability_bucket & mask_is_correct).sum()

    accuracy = bucket_correct_frequency / bucket_frequency

    bucket_probs = (t.arange(n_bins).to(toks.device) + 0.5) / n_bins
    variance = bucket_probs * (1 - bucket_probs) / bucket_frequency

    if plot_calibration_curve:
        y_upper = accuracy + variance.sqrt()
        y_lower = accuracy - variance.sqrt()
        my_line(accuracy.tolist(), y_upper.tolist(), y_lower.tolist(), title="Calibration curve")

    return (accuracy, variance, (bucket_frequency, bucket_correct_frequency)) if return_buckets else (accuracy, variance)



def plot_accuracies(
    accuracy_list: List[Float[Tensor, "n_bins"]],
    names: Optional[List[str]] = None,
    n_bins: int = 50,
    title: str = "Calibration curves",
):
    '''
    Plots accuracies (but with no variances).
    '''
    bucket_probs = (t.arange(n_bins) + 0.5) / n_bins

    fig = go.Figure()
    
    for accuracy in accuracy_list:
        fig.add_trace(go.Scatter(
            y=utils.to_numpy(accuracy),
            x=utils.to_numpy(bucket_probs),
            # line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name = names.pop(0) if names else None,
        ))
    fig.update_layout(
        width=800,
        height=600,
        # showlegend=False,
        title=title,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1]
    )
    fig.show()



def measure_calibration_all_heads(
    toks: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    minibatch_size: Optional[int] = None,
    top_k: int = 3,
    n_bins: int = 50,
    return_accuracies: bool = False,
):
    '''
    Measures the effect on the model's calibration metric of mean-ablating the direct effect of each attn head.

    Returns the results in a 2D tensor of shape [layers, heads] which can be displayed as heatmap.
    '''
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    bucket_frequency = t.zeros(1 + n_layers * n_heads, n_bins).to(toks.device)
    bucket_correct_frequency = t.zeros(1 + n_layers * n_heads, n_bins).to(toks.device)

    # Get the mean vectors we'll be using for ablations
    head_list = list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))
    result_mean = get_result_mean(
        head_list = head_list,
        toks = toks,
        model = model,
        keep_seq_dim = True,
        include_mlps = False,
        minibatch_size = minibatch_size,
        verbose = True,
    )

    toks_split: List[Tensor] = toks.split(minibatch_size) if minibatch_size is not None else [toks]

    for chunk_toks in tqdm(toks_split):

        gc.collect(); t.cuda.empty_cache()

        logits, cache = model.run_with_cache(
            chunk_toks,
            return_type="logits",
            names_filter = lambda name: name.endswith("result") or name == utils.get_act_name("scale"),
        )
        ln_final_scale = cache["scale"][:, :-1]
        
        next_toks = chunk_toks[:, 1:]

        for head_idx, head in enumerate([None] + head_list):
            
            # If we're measuring the baseline (no ablation) then we don't need to do anything
            if head is None:
                new_logits = logits[:, :-1]
            # Else, we get logits from removing the direct effect of this head
            else:
                L, H = head
                result = cache["result", L][:, :-1, H]
                result_ablated = result_mean[head].to(result.device)
                result_dla = (result - result_ablated[:-1]) / ln_final_scale
                new_logits: Tensor = logits[:, :-1] - (result_dla @ model.W_U)

            probs = new_logits.softmax(-1)
            topk_probs = probs.topk(k=top_k, dim=-1)
            topk_probs_values: Float[Tensor, "batch seq k"] = topk_probs.values
            topk_probs_indices: Int[Tensor, "batch seq k"] = topk_probs.indices
            
            # Figure out all the indices where the model was correct
            mask_is_correct: Bool[Tensor, "batch seq k"] = (topk_probs_indices == next_toks.unsqueeze(-1))

            for i in t.arange(n_bins):
                p_lower = i / n_bins
                p_upper = (i + 1) / n_bins
                # Figure out the indices of all values which are predicted for this probability
                mask_is_in_probability_bucket = (topk_probs_values >= p_lower) & (topk_probs_values < p_upper)
                # Calculate how many were correct
                bucket_frequency[head_idx, i] += mask_is_in_probability_bucket.sum()
                bucket_correct_frequency[head_idx, i] += (mask_is_in_probability_bucket & mask_is_correct).sum()


    accuracy = bucket_correct_frequency / bucket_frequency

    baseline_overconfidence = calculate_overconfidence(accuracy[0]).item()

    overconfidence_from_ablation = t.zeros(n_layers, n_heads)
    for head_idx, (L, H) in enumerate(head_list):
        overconfidence_from_ablation[L, H] = calculate_overconfidence(accuracy[head_idx+1]).item() / baseline_overconfidence - 1

    if return_accuracies:
        return overconfidence_from_ablation, {k: v for k, v in zip([None] + head_list, accuracy)}
    else:
        return overconfidence_from_ablation


# * old code for vectorised version of the t.arange(n_bins) line, don't think it's very helpful though, and it's super complicated

# mask_is_correct: Bool[Tensor, "batch seq k"] = (topk_probs_indices - next_toks) == 0

# t4 = time.time()
# prob_bins = einops.repeat(
#     t.arange(n_bins + 1, device=device) / n_bins,
#     "bins -> 1 1 1 bins"
# )
# t5 = time.time()

# mask_is_in_prob_bin: Bool[Tensor, "batch seq k bins"] = (topk_probs_values.unsqueeze(-1) >= prob_bins[..., :-1]) & (topk_probs_values.unsqueeze(-1) < prob_bins[..., 1:])
# bucket_frequency[i, :] += einops.reduce(mask_is_in_prob_bin, "batch seq k bins -> bins", "sum")
# bucket_correct_frequency[i, :] += einops.reduce(mask_is_in_prob_bin * mask_is_correct.unsqueeze(-1), "batch seq k bins -> bins", "sum")

# t6 = time.time()
# t_all["rest"] += t.tensor((t6 - t5, t5 - t4, t4 - t3, t3 - t2))
# progress_bar.update(1)