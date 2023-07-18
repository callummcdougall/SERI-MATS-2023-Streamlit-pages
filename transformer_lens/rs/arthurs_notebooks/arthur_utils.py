from transformer_lens.cautils.utils import *
import torch

def get_metric_from_end_state(
    model,
    end_state,
    targets = None,
    logits = None,
    return_logits = False,
    mode: Literal["loss", "kl"] = "loss",
    log_probs_reference: Optional[torch.Tensor] = None,
    device = None,
):
    # end state has shape batch, seq_len, hidden_size
    # targets has shape batch, seq_len

    assert (mode == "loss") != (log_probs_reference is not None), "Must specify kl_reference if mode is kl"
    assert (mode == "loss") == (targets is not None), "Must specify targets if mode is loss"

    if logits is None:
        if mode == "loss":
            assert list(end_state.shape) == list(targets.shape) + [
                model.cfg.d_model
            ], f"end_state.shape: {end_state.shape}, targets.shape: {targets.shape}"
        assert len(end_state.shape) == 3, "We stricter now"
        post_layer_norm = model.ln_final(end_state.to(device))
        logits = model.unembed(post_layer_norm)
    else:
        assert end_state is None
        assert logits.shape == targets.shape + (model.cfg.d_vocab,)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    if mode == "kl":
        assert log_probs_reference.shape == log_probs.shape

        total_kl = t.zeros((log_probs_reference.shape[0], log_probs_reference.shape[1]))
        for batch_idx in tqdm(range(log_probs_reference.shape[0])):
            gc.collect()
            torch.cuda.empty_cache()
            cur_kl = torch.nn.functional.kl_div(
                log_probs[batch_idx].to(device),
                log_probs_reference[batch_idx].to(device),
                log_target=True,
                reduction="none",
            ).sum(dim=-1).cpu()
            assert len(list(cur_kl.shape)) == 1, cur_kl.shape
            total_kl[batch_idx] = cur_kl
        
        assert len(total_kl.shape) == 2
        return total_kl

    if len(targets.shape) == 2:
        loss = -log_probs[
            torch.arange(targets.shape[0]).unsqueeze(1),
            torch.arange(targets.shape[1]).unsqueeze(0),
            targets,
        ]

    elif len(targets.shape) == 1:
        assert loss.shape[0]==1, loss.shape
        loss = -log_probs[
            :, torch.arange(targets.shape[0]), targets
        ]

    if return_logits:
        return loss, logits
    return loss

def get_filtered_webtext(model, batch_size=30, seed: int = 1729, device="cuda", max_seq_len=1024, dataset="stas/openwebtext-10k"):
    """
    Returns webtext that is all equal to length max token length. Ah.
    """
    dataset = get_webtext(seed=seed, dataset=dataset)
    filtered_tokens = []
    targets = []  # targets for prediction

    print("Not rapid, but not THAT slow :-) ")
    _idx = -1
    while len(filtered_tokens) < batch_size:
        _idx += 1
        cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
        if (
            len(cur_tokens) > max_seq_len # Greater Than so that we have all the targets for the context!!!
        ):  # so we're not biasing towards early sequence positions...
            filtered_tokens.append(cur_tokens[:max_seq_len])
            targets.append(cur_tokens[1 : max_seq_len + 1])

    mybatch = torch.LongTensor(filtered_tokens).to(device)
    mytargets = torch.LongTensor(targets).to(device)
    return mybatch, mytargets

def set_to_value(
    z, 
    hook,
    head_idx,
    new_value,
    seq_indices=None,
):
    if seq_indices is None:
        assert z[:, :, head_idx].shape == new_value.shape
        z[:, :, head_idx] = new_value
    else:
        assert len(seq_indices)==len(z)
        assert new_value.shape == (len(z), z.shape[-1])
        z[torch.arange(len(z)), seq_indices, head_idx] = new_value

    return z

def dot_with_query(
    unnormalized_keys: Float[torch.Tensor, "... d_model"],
    unnormalized_queries: Float[torch.Tensor, "... d_model"],
    model,
    layer_idx,
    head_idx,
    add_key_bias: bool = True, 
    add_query_bias: bool = True,
    normalize_keys: bool = True,
    normalize_queries: bool = True,
    use_tqdm: bool = True,
) -> Float[torch.Tensor, "..."]:

    warnings.warn("Please test this refactored version on the notebooks where you already used it")

    assert list(unnormalized_keys.shape) == list(unnormalized_queries.shape), (
        unnormalized_keys.shape,
        unnormalized_queries.shape,
    )

    W_Q = model.W_Q[layer_idx, head_idx]
    W_K = model.W_K[layer_idx, head_idx]

    if normalize_queries:
        queries = unnormalized_queries / (unnormalized_queries.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)
    else:
        queries = unnormalized_queries
    
    if normalize_keys:
        keys = unnormalized_keys / (unnormalized_keys.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)
    else:
        keys = unnormalized_keys

    results = torch.zeros(queries.shape[:-1])
    iterator = list(range(queries.shape[0]))
    iterator = tqdm(iterator) if use_tqdm else iterator
    for index_tuple in iterator: # TODO easy to batch, mate...
        q_vector, k_vector = queries[index_tuple], keys[index_tuple]
        gc.collect()
        torch.cuda.empty_cache()

        query_side_vectors = einops.einsum(
            q_vector,
            W_Q,
            "... d_model, ... d_model d_head -> ... d_head",
        ) 
        if add_query_bias:
            query_side_vectors += model.b_Q[layer_idx, head_idx]
        
        # TODO to do this addition maximally safe, assert some shapes and/or einops.repeat the bias
        key_side_vectors = einops.einsum(
            k_vector,
            W_K,
            "... d_model, ... d_model d_head -> ... d_head",
        )
        if add_key_bias:
            key_side_vectors += model.b_K[layer_idx, head_idx]

        assert list(query_side_vectors.shape) == list(key_side_vectors.shape), (
            query_side_vectors.shape,
            key_side_vectors.shape,
        )
        assert query_side_vectors.shape[-1] == key_side_vectors.shape[-1] == model.cfg.d_head, (
            query_side_vectors.shape,
            key_side_vectors.shape,
            model.cfg.d_head,
        )

        attention_scores = einops.einsum(
            query_side_vectors,
            key_side_vectors,
            "... d_head, ... d_head -> ...",
        ) / np.sqrt(model.cfg.d_head)
        results[index_tuple] = attention_scores

    return results