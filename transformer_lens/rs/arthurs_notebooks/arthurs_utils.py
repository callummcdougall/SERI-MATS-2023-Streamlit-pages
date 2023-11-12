
from transformer_lens.cautils.utils import *
import pytest

def get_metric_from_end_state(
    model: HookedTransformer,
    end_state: Optional[Float[torch.Tensor, "batch seq d_model"]] = None,
    targets: Optional[Int[torch.Tensor, "batch seq"]] = None,
    logits: Optional[Float[torch.Tensor, "batch seq d_vocab"]] = None,
    return_logits: bool = False,
    mode: Literal["loss", "kl"] = "loss",
    log_probs_reference: Optional[Float[torch.Tensor, "batch seq d_vocab"]] = None,
    frozen_ln_scale: Optional[Float[torch.Tensor, "batch seq 1"]] = None, # set to None to recompute the Layer Norm
    device: Optional[str] = None,
    compare_ln_scales: bool = False,
):
    """
    Compute the Lols (or KL Divergence from some reference)
    from the end state of the residual stream of a model
    """

    # if targets is None:
    #     warnings.warn("Don't have targets so I'm trimming the last sequence element")

    assert (mode == "loss") != (log_probs_reference is not None), "Must specify kl_reference if mode is kl"
    assert (mode == "loss") == (targets is not None), "Must specify targets if mode is loss"
    if frozen_ln_scale is not None:
        assert end_state is not None, "Recomputing LN only makes sense if we're recomputing from the end state"
        assert frozen_ln_scale.shape == (end_state.shape[0], end_state.shape[1], 1), frozen_ln_scale.shape

    if logits is None:
        if mode == "loss":
            assert list(end_state.shape) == list(targets.shape) + [
                model.cfg.d_model
            ], f"end_state.shape: {end_state.shape}, targets.shape: {targets.shape}"
        assert len(end_state.shape) == 3, "We stricter now"

        if frozen_ln_scale is None:
            post_layer_norm = model.ln_final(end_state.to(device))
                
        else:
            if compare_ln_scales:
                cache = {}
                model.cache_some(names = lambda name: name=="ln_final.hook_scale", cache=cache)
                fake_layer_norm = model.ln_final(end_state.to(device).clone())

            post_layer_norm = end_state.to(device) / frozen_ln_scale

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
        warnings.warn("Uh, this looks somewhat sketchy - check?")
        assert loss.shape[0]==1, loss.shape
        loss = -log_probs[
            :, torch.arange(targets.shape[0]), targets
        ]

    if compare_ln_scales:
        return loss, cache["ln_final.hook_scale"]
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
        if hook.name.endswith(("scores", "pattern")):
            assert z[:, head_idx].shape == new_value.shape
            z[:, head_idx] = new_value

        else:
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
    use_tqdm: bool = False,
) -> Float[torch.Tensor, "..."]:
    
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

@pytest.mark.parametrize("freeze_ln", [True, False]) # test the versions that have both frozen and unfrozen LN
def test_get_metric_from_end_state(freeze_ln):
    """Cribbed from direct_effect_survey.py"""

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=False,
    )
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_attn_in(True)
    DEVICE = "cuda"
    SHOW_PLOT = True
    BATCH_SIZE = 25

    mybatch, mytargets = get_filtered_webtext(model, batch_size=BATCH_SIZE)
    NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
    END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
    ln_scale_hook_name = "ln_final.hook_scale"
    names_filter1 = (
        lambda name: name == END_STATE_HOOK
        or name.endswith("hook_result")
        or name.endswith(".hook_resid_pre")
        or name == get_act_name("resid_mid", NEGATIVE_LAYER_IDX)
        or name == get_act_name("resid_pre", NEGATIVE_LAYER_IDX+1)
        or name == get_act_name("resid_mid", NEGATIVE_LAYER_IDX+1)
        or name == ln_scale_hook_name
    )
    logits, cache = model.run_with_cache(
        mybatch.to("cuda"),
        names_filter=names_filter1,
    )
    end_state = cache[END_STATE_HOOK].to("cuda")
    full_log_probs = torch.nn.functional.log_softmax(logits.cuda(), dim=-1).cpu()
    del logits
    gc.collect()
    torch.cuda.empty_cache()

    my_loss = get_metric_from_end_state(model, end_state.to(DEVICE), mytargets, frozen_ln_scale = None if not freeze_ln else cache[ln_scale_hook_name]).cpu()

    their_loss = model(
        mybatch.to(DEVICE),
        return_type="loss",
        loss_per_token=True,
    ).cpu()
    assert list(their_loss.shape) == [
        my_loss.shape[0],
        my_loss.shape[1] - 1,
    ], f"their_loss.shape: {their_loss.shape}, my_loss.shape: {my_loss.shape}"

    torch.testing.assert_close(
        their_loss,
        my_loss[:, :-1],
        atol=1e-2,
        rtol=1e-2,
    ) # yey

def get_batch_tokens(
    lm, 
    dataset: Iterator[Dict], # Dict should contain "text": "This is my batch element...". In practice take HF dataset and apply iter()
    batch_size: int,
    seq_len: int,
    device: Optional[torch.device] = None, # Defaults to `lm.cfg.device`
    use_tqdm: bool = False,
):
    """Stolen from my SAE repo"""

    batch_tokens = torch.LongTensor(size=(0, seq_len)).to(device or lm.cfg.device)
    current_batch = []
    current_length = 0

    if use_tqdm:
        pbar = tqdm(total=batch_size, desc="Filling batches")
    
    while batch_tokens.shape[0] < batch_size:
        s = next(dataset)["text"]
        tokens = lm.to_tokens(s, truncate=False, move_to_device=True).squeeze(0)
        assert len(tokens.shape) == 1, f"tokens.shape should be 1D but was {tokens.shape}"
        token_len = tokens.shape[0]
    
        while token_len > 0:
            # Space left in the current batch
            space_left = seq_len - current_length
    
            # If the current tokens fit entirely into the remaining space
            if token_len <= space_left:
                current_batch.append(tokens[:token_len])
                current_length += token_len
                break
    
            else:
                # Take as much as will fit
                current_batch.append(tokens[:space_left])
    
                # Remove used part, add BOS
                tokens = tokens[space_left:]
                tokens = torch.cat((torch.LongTensor([lm.tokenizer.bos_token_id]).to(tokens.device), tokens), dim=0)
    
                token_len -= space_left
                token_len += 1
                current_length = seq_len
    
            # If a batch is full, concatenate and move to next batch
            if current_length == seq_len:
                full_batch = torch.cat(current_batch, dim=0)
                batch_tokens = torch.cat((batch_tokens, full_batch.unsqueeze(0)), dim=0)
                current_batch = []
                current_length = 0

        if use_tqdm:    
            pbar.n = batch_tokens.shape[0]
            pbar.refresh()
    
    return batch_tokens[:batch_size]