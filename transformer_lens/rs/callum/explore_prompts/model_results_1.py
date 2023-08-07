# This file contains functions to get results from my model. I use them to generate the visualisations found on the Streamlit pages.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
from typing import Dict, Any, Tuple, List, Optional, Literal
from transformer_lens import HookedTransformer, utils
from functools import partial
import einops
from dataclasses import dataclass
from transformer_lens.hook_points import HookPoint
import pickle
from jaxtyping import Int, Float, Bool
import torch as t
from collections import defaultdict
from torch import Tensor

Head = Tuple[int, int]

FUNCTION_STR_TOKS =  [
    '\x00',
    '\x01',
    '\x02',
    '\x03',
    '\x04',
    '\x05',
    '\x06',
    '\x07',
    '\x08',
    '\t',
    '\n',
    '\x0b',
    '\x0c',
    '\r',
    '\x0e',
    '\x0f',
    '\x10',
    '\x11',
    '\x12',
    '\x13',
    '\x14',
    '\x15',
    '\x16',
    '\x17',
    '\x18',
    '\x19',
    '\x1a',
    '\x1b',
    '\x1c',
    '\x1d',
    '\x1e',
    '\x1f',
    '\x7f',
    ' a',
    ' the',
    ' an',
    ' to',
    ' of',
    ' in',
    ' that',
    ' for',
    ' it',
    ' with',
    ' as',
    ' was',
    ' The',
    ' are',
    ' by',
    ' have',
    ' this',
    'The',
    ' will',
    ' they',
    ' their',
    ' which',
    ' about',
    ' In',
    ' like',
    ' them',
    ' some',
    ' when',
    ' It',
    ' what',
    ' its',
    ' only',
    ' how',
    ' most',
    ' This',
    ' these',
    ' very',
    ' much',
    ' those',
    ' such',
    ' But',
    ' You',
    ' If',
    ' take',
    'It',
    ' As',
    ' For',
    ' They',
    'the',
    ' That',
    'But',
    ' When',
    ' To',
    'As',
    ' almost',
    ' With',
    ' However',
    'When',
    ' These',
    'That',
    'To',
    ' By',
    ' takes',
    ' While',
    ' whose',
    'With',
    ' Of',
    ' THE',
    ' From',
    ' aren',
    'While',
    ' perhaps',
    'By',
    ' whatever',
    'with',
    ' Like',
    'However',
    'Of',
    ' Their',
    ' Those',
    ' Its',
    ' Thus',
    ' Such',
    'Those',
    ' Much',
    ' Between',
    'Their',
    ' meanwhile',
    ' nevertheless',
    'Its',
    ' at', ' of', 'to', ' now', "'s", 'The', ".", ",", # Need to go through these words and add more to them, I suspect this list is minimal
]


class HeadResults:
    data: Dict[Head, Any]
    '''
    Stores results for each head. This is just a standard dictionary, but you can index it like [layer, head]
    (this is just a convenience thing, so I don't have to wrap the head in a tuple).
    '''
    def __init__(self, data=None):
        self.data = data or {}
    def __getitem__(self, layer_and_head: Head) -> Any:
        return self.data[layer_and_head]
    def __setitem__(self, layer_and_head: Head, value):
        self.data[layer_and_head] = value
    def items(self):
        return self.data.items()


class LayerResults:
    data: Dict[int, Any]
    '''
    Similar syntax to HeadResults, but the keys are ints (layers) not (layer, head) tuples.
    '''
    def __init__(self, data=None):
        self.data = data or {}
    def __getitem__(self, layer: int) -> Any:
        return self.data[layer]
    def __setitem__(self, layer: int, value):
        self.data[layer] = value
    def items(self):
        return self.data.items()


class DictOfHeadResults:
    data: Dict[int, HeadResults]
    '''
    This is useful when e.g. I have a bunch of different patching modes:
        effect = (direct/indirect/both)
        ln_mode = (frozen/unfrozen)
        ablation_mode = (zero/mean)
    and I want to store a HeadResults object for each mode.
    '''
    def __init__(self, data=None):
        self.data = data or defaultdict(lambda: HeadResults())
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value
    def items(self):
        return self.data.items()


class ModelResults:
    '''
    All the datatypes are one of 3 things:

    1. Just plain old tensors

    2. HeadResults: this is nominally a dictionary mapping (layer, head) tuples to values,
       the only difference is you can index it with [layer, head], like you can for tensors.

    3. DictOfHeadResults: this is when I want to store a dictionary mapping strings to
       HeadResults. For example, I want to store many different types of loss for a head,
       and I want to store them all in one place rather than having many different attributes.
    '''

    def __init__(self):
        
        # Original loss and logits (no interventions)
        self.logits_orig: Tensor = t.empty(0)
        self.loss_orig: Tensor = t.empty(0)

        # All logits and loss, for the (effects=3 * ln_modes=2 * ablation_modes=2) = 12 types of ablation
        self.logits: DictOfHeadResults = DictOfHeadResults()
        self.dla: DictOfHeadResults = DictOfHeadResults()
        self.loss_diffs: DictOfHeadResults = DictOfHeadResults()

        # An array indicating whether each token is an instance of copy-suppression
        self.is_copy_suppression: DictOfHeadResults = DictOfHeadResults()

        # We need the result vectors for each head, we'll use it for patching
        self.result: HeadResults = HeadResults()
        self.result_mean: HeadResults = HeadResults()

        # We need data for attn & weighted attn (and for projecting result at source token unembeds)
        self.pattern: HeadResults = HeadResults()
        self.v: HeadResults = HeadResults()
        self.out_norm: HeadResults = HeadResults()

        # resid_pre for each head (to get unembed components), and resid_post the very end
        self.resid_pre: LayerResults = LayerResults()
        self.resid_post: LayerResults = LayerResults()
        
        # Layernorm scaling factors, pre-attention layer for each head & at the very end
        self.scale_attn: LayerResults = LayerResults()
        self.scale: Tensor = t.empty(0)

        # This is the "prediction-attention" part
        self.unembedding_components: LayerResults = LayerResults()

    def clear(self):
        """Empties all imtermediate results we don't need."""
        del self.result
        del self.result_mean
        del self.resid_pre
        del self.resid_post
        del self.v
        del self.scale_attn

    def items(self):
        return self.__dict__.items()




def first_occurrence(input_list):
    '''Thanks ChatGPT!'''
    seen = {}
    return [seen.setdefault(x, True) and not seen.update({x: False}) for x in input_list]



def gram_schmidt(vectors: Float[Tensor, "... d num"]) -> Float[Tensor, "... d num"]:
    '''
    Performs Gram-Schmidt orthonormalization on a batch of vectors, returning a basis.

    `vectors` is a batch of vectors. If it was 2D, then it would be `num` vectors each with length
    `d`, and I'd want a basis for these vectors in d-dimensional space. If it has more dimensions 
    at the start, then I want to do the same thing for all of them (i.e. get multiple independent
    bases).
    '''
    # Make a copy of the vectors
    basis = vectors.clone()
    num_vectors = basis.shape[-1]
    
    # Iterate over each vector in the batch, starting from the zeroth
    for i in range(num_vectors):
        # Project the i-th vector onto the space orthogonal to the previous ones
        for j in range(i):
            projection = einops.einsum(basis[..., i], basis[..., j], "... d, ... d -> ...")
            basis[..., i] = basis[..., i] - einops.einsum(projection, basis[..., j], "..., ... d -> ... d")
        
        # Normalize this vector
        basis[..., i] = basis[..., i] / t.norm(basis[..., i], dim=-1, keepdim=True)
    
    return basis



def project(
    vectors: Float[Tensor, "... d"],
    proj_directions: Float[Tensor, "... d num"],
    only_keep: Optional[Literal["pos", "neg"]] = None
):
    '''
    `vectors` is a batch of vectors, with last dimension `d` and all earlier dimensions as batch dims.

    `proj_directions` is either the same shape as `vectors`, or has an extra dim at the end.

    If they have the same shape, we project each vector in `vectors` onto the corresponding direction
    in `proj_directions`. If `proj_directions` has an extra dim, then the last dimension is another 
    batch dim, i.e. we're projecting each vector onto a subspace rather than a single vector.
    '''
    assert proj_directions.shape[:-1] == vectors.shape
    assert proj_directions.shape[-1] <= 30, "Shouldn't do too many vectors, GS orth might be computationally heavy I think?"

    proj_directions_basis = gram_schmidt(proj_directions)

    components_in_proj_dir = einops.einsum(
        vectors, proj_directions_basis,
        "... d, ... d num -> ... num"
    )
    
    if only_keep is not None:
        components_in_proj_dir = t.where(
            (components_in_proj_dir < 0) if (only_keep == "neg") else (components_in_proj_dir > 0),
            components_in_proj_dir,
            t.zeros_like(components_in_proj_dir)
        )

    vectors_projected = einops.einsum(
        components_in_proj_dir,
        proj_directions_basis,
        "... num, ... d num -> ... d"
    )

    return vectors_projected


def model_fwd_pass_from_resid_pre(
    model: HookedTransformer, 
    resid_pre: Float[Tensor, "batch seq d_model"],
    layer: int
) -> Float[Tensor, "batch seq d_vocab"]:
    '''
    Performs a forward pass starting from an intermediate point.

    For instance, if layer=10, this will apply the TransformerBlocks from
    layers 10, 11 respectively, then ln_final and unembed.
    '''
    resid = resid_pre
    for i in range(layer, model.cfg.n_layers):
        resid = model.blocks[i](resid)
    
    resid = model.ln_final(resid)
    logits = model.unembed(resid)
    return logits


def get_model_results(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_heads: List[Tuple[int, int]],
    use_cuda: bool = False,
    verbose: bool = False,
    num_top_src_tokens: int = 3,
    num_top_CS_tokens: Optional[int] = None,
    loss_ratio_threshold: float = 0.6,
) -> ModelResults:
    '''
    Explanation of how the last 3 parameters determine the copy-suppression classification:

    The intervention we perform to calculate "ablation, preserving copy-suppression" is as follows:

        (1) Get the output vectors (v @ W_O) for the attention head, and multiply them all by the
            attention probs - so we have something of shape (batch, seqQ, seqK, d_model).

        (2) Pick the top `num_top_src_tokens` source tokens for each destination token, i.e. the ones
            whose unembeddings are most present in the query-side residual stream input.

        (3) Project these vectors onto directions D (see below), and mean ablate the rest. Note that
            the projection is also done wrt the mean.
        
        (4) Calculate the new logits (we can use direct/indirect/both effects).

        (5) Calculate the new loss, which we'll call loss_CS. This represents the loss when we ablate
            everything except for the CS mechanism. We also have loss_orig and loss_ablated.

        (6) Classify this as copy-suppression if (loss_ablated - loss_CS) / (loss_ablated - loss_orig)
            is sufficiently close to 1 (because this means that deleting everything except for the CS
            mechanism is still quite close to original performance).

    What directions D do we use? Initially, I thought we should just project the vector at each source
    token onto its unembedding. But this ignores the fact that copy-suppression can do other things, 
    like "a token attends to ' Pier', and ' pier' is suppressed." So if `num_top_CS_tokens` is not None,
    then rather than projecting onto the unembedding of the source token, we project onto this many of
    the unembeddings of the most negatively copied tokens (i.e. the ones with the most negative values
    in the row of W_EE @ W_OV @ W_U which corresponds to the source token). W_EE is our extended 
    embedding, i.e. the thing we get from including W_E, Attn0 (fixed to self-attn) and MLP0.
    '''

    assert num_top_src_tokens <= 10, "Can't do more than 10 tokens."
    FUNCTION_TOKS = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze()

    if verbose: print("Computing unembeddings for copy-suppression classifications...")
    
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()

    current_device = str(next(iter(model.parameters())).device)
    if use_cuda and current_device == "cpu":
        model = model.cuda()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cpu()

    model_results = ModelResults()

    unembeddings: Float[Tensor, "batch seq d_model"] = model.W_U.T[toks]
    # unembeddings_normed = unembeddings / t.norm(unembeddings, dim=-1, keepdim=True)

    # * Here is we get the thing that we'll be projecting onto, as part of our copy-suppression classifications.
    # * See the docstring for more info.
    unembeddings_CS = HeadResults()
    if num_top_CS_tokens is None:
        for (layer, head) in negative_heads:
            unembeddings_CS[layer, head]: Float[Tensor, "batch seq d_model 1"] = unembeddings.unsqueeze(-1)
    else:
        from transformer_lens.rs.callum.keys_fixed import get_effective_embedding_2
        W_EE_dict = get_effective_embedding_2(model)
        W_EE = W_EE_dict["W_E (including MLPs)"]
        W_U = W_EE_dict["W_U"].T
        for layer, head in negative_heads:
            W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
            W_EE_OV: Float[Tensor, "batch seq d_model"] = W_EE[toks] @ W_OV
            W_EE_OV_scaled = W_EE_OV / W_EE_OV.std(dim=-1, keepdim=True)
            W_OV_full: Float[Tensor, "batch seq d_vocab"] = W_EE_OV_scaled @ W_U
            most_copy_suppressed_toks: Int[Tensor, "batch seq k"] = W_OV_full.topk(dim=-1, k=num_top_CS_tokens, largest=False).indices
            unembeddings_CS[(layer, head)]: Float[Tensor, "batch seq d_model k"] = model.W_U.T[most_copy_suppressed_toks].transpose(-1, -2)

    if verbose: print("Running forward pass...")

    # Cache the head results and attention patterns, and final ln scale, and residual stream pre heads
    # (note, this is preferable to using names_filter argument for cache, because you can't pick specific heads)

    def cache_head_result(result: Float[Tensor, "batch seq n_heads d_model"], hook: HookPoint, head: int):
        model_results.result[hook.layer(), head] = result[:, :, head]
    
    def cache_head_pattern(pattern: Float[Tensor, "batch n_heads seqQ seqK"], hook: HookPoint, head: int):
        model_results.pattern[hook.layer(), head] = pattern[:, head]
    
    def cache_head_v(v: Float[Tensor, "batch seq n_heads d_head"], hook: HookPoint, head: int):
        model_results.v[hook.layer(), head] = v[:, :, head]
    
    def cache_scale(scale: Float[Tensor, "batch seq 1"], hook: HookPoint):
        model_results.scale = scale

    def cache_resid_pre(resid_pre: Float[Tensor, "batch seq d_model"], hook: HookPoint):
        model_results.resid_pre[hook.layer()] = resid_pre

    def cache_resid_post(resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint, key: Any):
        hook.ctx[key] = resid_post

    def cache_scale_attn(scale: Float[Tensor, "batch seq 1"], hook: HookPoint):
        model_results.scale_attn[hook.layer()] = scale
    

    # To calculate the unembedding components, we need values per layer rather than per head
    all_layers = sorted(set([layer for layer, head in negative_heads]))
    for layer in all_layers:
        model.add_hook(utils.get_act_name("resid_pre", layer), cache_resid_pre)
        model.add_hook(utils.get_act_name("scale", layer, "ln1"), cache_scale_attn)
    # For most other things, we do need values per head
    for layer, head in negative_heads:
        model.add_hook(utils.get_act_name("result", layer), partial(cache_head_result, head=head))
        model.add_hook(utils.get_act_name("v", layer), partial(cache_head_v, head=head))
        model.add_hook(utils.get_act_name("pattern", layer), partial(cache_head_pattern, head=head))
    # We also need some things at the very end of the model
    model.add_hook(utils.get_act_name("resid_post", model.cfg.n_layers-1), partial(cache_resid_post, key="orig"))
    model.add_hook(utils.get_act_name("scale"), cache_scale)

    # Run the forward pass, to cache all values (and get original logits)
    model_results.logits_orig, model_results.loss_orig = model(toks, return_type="both", loss_per_token=True)
    model.reset_hooks(clear_contexts=False)

    # Calculate the unembedding components stored in the residual stream, for each word in context

    if verbose: print("Computing unembedding components...")

    for layer in all_layers:
        resid_pre_scaled: Float[Tensor, "batch seq d_model"] = model_results.resid_pre[layer] / model_results.scale_attn[layer]
        assert t.all((resid_pre_scaled.norm(dim=-1) - model.cfg.d_model ** 0.5) < 1e-4)
        resid_pre_normed = resid_pre_scaled / t.norm(resid_pre_scaled, dim=-1, keepdim=True)

        # Define a mask which is both causal (q >= k) and only keeps the first occurrence of each source token (no repeats in the table!)
        batch_size, seq_len = resid_pre_normed.shape[:2]
        k = min(10, seq_len)
        q_indices = einops.repeat(t.arange(seq_len), "seqQ -> seqQ seqK", seqK=seq_len)
        k_indices = einops.repeat(t.arange(seq_len), "seqK -> seqQ seqK", seqQ=seq_len)
        causal_mask = einops.repeat(q_indices >= k_indices, "seqQ seqK -> batch seqQ seqK", batch=batch_size)
        first_occurrence_mask: Bool[Tensor, "batch seqQ seqK"] = einops.repeat(
            t.stack([t.tensor(first_occurrence(seq.tolist())) for seq in toks]),
            "batch seqK -> batch seqQ seqK", seqQ=seq_len
        )
        function_word_mask = einops.repeat(
            (toks.unsqueeze(-1) == einops.repeat(FUNCTION_TOKS, "t -> 1 1 t")).any(-1),
            "batch seqK -> batch seqQ seqK", batch=batch_size, seqQ=seq_len
        )
        mask = causal_mask & first_occurrence_mask
        mask_fw = causal_mask & first_occurrence_mask & (~function_word_mask)
        # causal_mask_rm_self = (q_indices >= k_indices) & (q_tokens != k_tokens)

        unembedding_components: Float[Tensor, "batch seq d_model"] = einops.einsum(
            resid_pre_normed, unembeddings,
            "batch seqQ d_model, batch seqK d_model -> batch seqQ seqK"
        )
        unembedding_components_avg = (unembedding_components * mask).sum(dim=-1) / mask.sum(dim=-1)
        unembedding_components_avg_fw = (unembedding_components * mask_fw).sum(dim=-1) / mask_fw.sum(dim=-1)
        unembedding_components_top10 = t.where(mask, unembedding_components, -1.0e9).topk(k, dim=-1)
        unembedding_components_top10_fw = t.where(mask_fw, unembedding_components, -1.0e9).topk(k, dim=-1)
        
        model_results.unembedding_components[layer] = {
            ("avg", "inc_fw"): unembedding_components_avg,
            ("avg", "not_inc_fw"): unembedding_components_avg_fw,
            ("top10", "inc_fw"): unembedding_components_top10,
            ("top10", "not_inc_fw"): unembedding_components_top10_fw,
        }

    for (layer, head) in negative_heads:
        
        # Get output norms for value-weighted attention
        out = einops.einsum(
            model_results.v[layer, head], model.W_O[layer, head],
            "batch seq d_head, d_head d_model -> batch seq d_model"
        )
        out_norm = einops.reduce(out.pow(2), "batch seq d_model -> batch seq", "sum").sqrt()
        model_results.out_norm[layer, head] = out_norm

        # Calculate the thing we'll be subbing in for mean ablation
        model_results.result_mean[layer, head] = einops.reduce(
            model_results.result[layer, head], 
            "batch seq d_model -> d_model", "mean"
        )

    if verbose: print("Computing logits, loss and direct effects...")

    # ! Now for the big ones: the logits, loss, and direct effects
    # For each head, there are 12 different logit terms: 3 effects (direct/indirect/both), 2 ablation modes (mean/zero), 2 final layernorm modes (frozen/unfrozen)
    # I need to get the final value of the residual stream (pre-LN) for each of the 3*2*1 = 6 modes, then I can directly compute all 12 using model_results.scale / normalization
    # I calculate "both" by doing patching at the output of the head, "direct" by patching at resid_post final, and "indirect" by subtracting the two
    # What about DLA? For each head, there are 4 different kinds (mean/zero, frozen/unfrozen), and so I also store these at the same time
    # What about loss? We have 12 different losses, one for each of the 12 logit terms.

    resid_post_orig = model.hook_dict[utils.get_act_name("resid_post", model.cfg.n_layers-1)].ctx.pop("orig")

    def ablate_head_result(
        result: Float[Tensor, "batch seq n_heads d_model"],
        hook: HookPoint,
        head: int,
        ablation_type: Literal["mean", "zero"],
    ):
        '''
        Simple form of ablation - we just replace the result at a particular head with the ablated values (either zero or
        the thing supplied by `ablation_values`).
        '''
        assert ablation_type in ["mean", "zero"]

        if ablation_type == "zero":
            result[:, :, head] = t.zeros_like(result[:, :, head])
        else:
            result[:, :, head] = model_results.result_mean[hook.layer(), head]

        # Store in hook context, and return
        hook.ctx[("result_ablated", head)] = result[:, :, head].clone()
        return result


    def ablate_head_result_preserve_copy_suppression_mechanism(
        result: Float[Tensor, "batch seq n_heads d_model"],
        hook: HookPoint,
        head: int,
        ablation_type: Literal["mean", "zero"],
        num_top_src_tokens: int = 3,
    ):
        '''
        Much more complex form of ablation - we replace all pre-attn result vectors (v @ W_O) with their ablated values,
        except for the ones whose unembeddings are among the largest `top_src_tokens` for the corresponding query-side
        residual stream vector. Those ones, we preserve.
        
        We also store the ablated values in context (so that we can use them later to compute direct effects).

        The actual value we replace the attention head with is just the ablated values.
        '''
        assert ablation_type in ["mean", "zero"]
        batch_size, seq_len = result.shape[:2]
        layer = hook.layer()

        # Get the results pre attn (i.e. v @ W_O), and also get the ablated results.
        # Note, we ablate after multiplying by attention rather than before. This is because ablating
        # should remove all signals: attention (QK) and values (OV).
        result_pre_attn: Float[Tensor, "batch seqK d_model"] = model_results.v[layer, head] @ model.W_O[layer, head]
        result_post_attn: Float[Tensor, "batch seqQ seqK d_model"] = einops.einsum(
            result_pre_attn,
            model_results.pattern[layer, head],
            "batch seqK d_model, batch seqQ seqK -> batch seqQ seqK d_model"
        )
        result_post_attn_ablated = result_post_attn.clone()
        result_post_attn_ablated[:] = einops.reduce(
            result_post_attn_ablated,
            "batch seqQ seqK d_model -> d_model",
            "mean"
        ) if (ablation_type=="mean") else 0.0

        # Compute the projections of the (v @ W_O) terms.
        # Note how we subtract the ablated results before projecting, e.g. if we're doing mean ablation
        # then we only want to project the difference between this vector and its mean. 
        # Also note how it does make a difference whether we project pre_attn or post_attn (because 
        # we're projecting wrt the ablation, and the ablation is defined wrt post_attn).
        unembeddings_repeated = einops.repeat(
            unembeddings_CS[layer, head],
            "batch seqK d_model k -> batch seqQ seqK d_model k",
            seqQ = seq_len,
        )
        result_post_attn_projections = result_post_attn.clone()
        result_post_attn_projections = project(
            vectors = result_post_attn_projections - result_post_attn_ablated,
            proj_directions = unembeddings_repeated,
            # only_keep = "neg",
        ) + result_post_attn_ablated

        # Compute the places we'll be projecting at
        unembedding_components_top10 = model_results.unembedding_components[layer][("top10", "not_inc_fw")]
        # 'topk_src_tokens_indices' contains the indices of all source tokens we want to keep, for each (sequence, destination token).
        topk_src_tokens_indices: Int[Tensor, "batch seqQ top_seqK"] = unembedding_components_top10.indices[..., :num_top_src_tokens]
        # 'topk_src_tokens' contains the actual tokens corresopnding to these indices which we want to keep
        index = einops.repeat(t.arange(batch_size), "batch -> batch seqQ top_seqK", seqQ=seq_len, top_seqK=num_top_src_tokens)
        topk_src_tokens: Int[Tensor, "batch seqQ top_seqK"] = toks[index, topk_src_tokens_indices]
        # Now, make sure all the places where the values are actually set to -1e9 small are filtered out
        topk_src_tokens = t.where(
            unembedding_components_top10.values[..., :num_top_src_tokens] > -1e4,
            topk_src_tokens,
            t.full_like(topk_src_tokens, -1),
        )
        # 'keep_toks' is a boolean telling us which tokens will be kept (i.e. which ones match at any of the topk_src_tokens)
        toks_repeated = einops.repeat(toks, "batch seqK -> batch seqQ top_seqK seqK", seqQ=seq_len, top_seqK=num_top_src_tokens)
        topk_src_tokens_repeated = einops.repeat(topk_src_tokens, "batch seqQ top_seqK -> batch seqQ top_seqK seqK", seqK=seq_len)
        keep_toks: Bool[Tensor, "batch seqQ seqK"] = (toks_repeated == topk_src_tokens_repeated).any(dim=-2)

        # print(keep_toks.int().sum(-1))

        # Finally, get our new values for 'result_post_attn', which are either projections or ablated values everywhere (and the projections
        # are with respect to the ablated values).
        result_post_attn_new = t.where(
            keep_toks.unsqueeze(-1),
            result_post_attn_projections, # replace top src tokens with their projections
            result_post_attn_ablated # replace everything else with ablated values
        )
        result_new = einops.reduce(result_post_attn_new, "batch seqQ seqK d_model -> batch seqQ d_model", "sum")

        # Store in hook context
        hook.ctx[("result_ablated_CS", head)] = result_new.clone()

        # Ablate the head the normal way (for the indirect effect)
        if ablation_type == "zero":
            result[:, :, head] = t.zeros_like(result[:, :, head])
        else:
            result[:, :, head] = model_results.result_mean[hook.layer(), head]

        return result

    """
    Returns an array of shape (batch, seq_len), where each element is 1 if this is an example of
    copy-suppression, and 0 if not. We also have the option 2 if the experiment is inconclusive.

    We define copy-suppression (with somewhat arbitrary choice of metrics) for each destination token as follows:

        (1) Take the top `topk_src_tokens` source tokens on the query side, i.e. those s.t. the component of their
            unembeddings in the query-side residual stream before head 10.7 is the largest (excluding BOS).
        
        (2) Take the result vectors from those source tokens, and project them onto the direction of those source
            tokens' unembeddings (only keeping components which are negative).

        (3) Compute the (loss_ablated - loss_ablated_CS_preserved) / (loss_ablated - loss_orig). If this value is larger than
            loss_threshold, then we say that this is an example of copy-suppression.
    
    Explanation for step (3) - if this value is close to 1 (e.g. in situations where the head is beneficial, i.e. 
    ablating it increases the loss), it means that just taking this projection of the head's output is enough to 
    preserve most of the head's performance. So most of the way in which this head is affecting the loss is through
    the copy-suppression mechanism.

    We return 2 (inconclusive) in any of the following cases:
        (A) This datapoint is not in the top or bottom 5% of abs(loss_ablated - loss_orig) examples, since we only care
            about the cases where the head meaningfully affects model performance.
        (B) This datapoint doesn't have `topk_src_tokens` source tokens, e.g. because it's one of the first 10 tokens
            in the sequence.
    """

    # For each head:
    for layer, head in negative_heads:

        # For each ablation mode:
        for ablation_type in ["mean", "zero"]:

            # ! Calculate new logits, for the "both" intervention effect

            # Calculate "both" for unfrozen layernorm, by replacing this head's output with the ablated version and doing forward pass
            # Also calculate "both" for frozen layernorm, by directly computing it from the ablated value of the residual stream (which we get from patching at attn head result)
            model.add_hook(utils.get_act_name("result", layer), partial(ablate_head_result, head=head, ablation_type=ablation_type))
            model.add_hook(utils.get_act_name("resid_post", model.cfg.n_layers-1), partial(cache_resid_post, key="both"))
            model_results.logits[("both", "unfrozen", ablation_type)][layer, head] = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer)
            resid_post_both_ablated = model.hook_dict[utils.get_act_name("resid_post", model.cfg.n_layers-1)].ctx.pop("both")
            model_results.logits[("both", "frozen", ablation_type)][layer, head] = (resid_post_both_ablated / model_results.scale) @ model.W_U + model.b_U

            # Do exactly the same thing, just with a different (fancier!) form of ablation which preserves the pure copy-suppression mechanism.
            model.add_hook(utils.get_act_name("result", layer), partial(ablate_head_result_preserve_copy_suppression_mechanism, head=head, ablation_type=ablation_type, num_top_src_tokens=num_top_src_tokens))
            model.add_hook(utils.get_act_name("resid_post", model.cfg.n_layers-1), partial(cache_resid_post, key="both"))
            model_results.logits[("both", "unfrozen", ablation_type, "CS")][layer, head] = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer)
            resid_post_both_ablated_CS = model.hook_dict[utils.get_act_name("resid_post", model.cfg.n_layers-1)].ctx.pop("both")
            model_results.logits[("both", "frozen", ablation_type, "CS")][layer, head] = (resid_post_both_ablated_CS / model_results.scale) @ model.W_U + model.b_U

            # ! Calculate new logits & C-S classification, for the "direct" intervention effect, and calculate the DLA

            # Calculate "direct" for frozen layernorm, directly from the new value of the residual stream (resid_post_direct_ablated)
            # Also calculate "direct" for unfrozen layernorm, again directly from this new value of the residual stream
            resid_post_DLA = model_results.result[layer, head] - model.hook_dict[utils.get_act_name("result", layer)].ctx.pop(("result_ablated", head))
            resid_post_direct_ablated = resid_post_orig - resid_post_DLA
            logits_direct_ablated_frozen = (resid_post_direct_ablated / model_results.scale) @ model.W_U + model.b_U
            logits_direct_ablated_unfrozen = (resid_post_direct_ablated / resid_post_direct_ablated.std(-1, keepdim=True)) @ model.W_U + model.b_U
            model_results.logits[("direct", "frozen", ablation_type)][layer, head] = logits_direct_ablated_frozen
            model_results.logits[("direct", "unfrozen", ablation_type)][layer, head] = logits_direct_ablated_unfrozen

            # Do exactly the same thing, just with a different (fancier!) form of ablation which preserves the pure copy-suppression mechanism.
            resid_post_DLA_CS = model_results.result[layer, head] - model.hook_dict[utils.get_act_name("result", layer)].ctx.pop(("result_ablated_CS", head))
            resid_post_direct_ablated_CS = resid_post_orig - resid_post_DLA_CS
            logits_direct_ablated_frozen_CS = (resid_post_direct_ablated_CS / model_results.scale) @ model.W_U + model.b_U
            logits_direct_ablated_unfrozen_CS = (resid_post_direct_ablated_CS / resid_post_direct_ablated_CS.std(-1, keepdim=True)) @ model.W_U + model.b_U
            model_results.logits[("direct", "frozen", ablation_type, "CS")][layer, head] = logits_direct_ablated_frozen_CS
            model_results.logits[("direct", "unfrozen", ablation_type, "CS")][layer, head] = logits_direct_ablated_unfrozen_CS

            # This is also where we compute the direct logit attribution for this head
            model_results.dla[("frozen", ablation_type)][layer, head] = (resid_post_DLA / model_results.scale) @ model.W_U
            model_results.dla[("unfrozen", ablation_type)][layer, head] = model_results.logits_orig - logits_direct_ablated_unfrozen

            # ! Calculate new logits & C-S classification, for the "indirect" intervention effect

            # Calculate "indirect" for frozen layernorm, directly from the new value of the residual stream (resid_post_indirect_ablated)
            # Also calculate "indirect" for unfrozen layernorm, again directly from this new value of the residual stream
            resid_post_indirect_ablated = resid_post_orig + (resid_post_both_ablated - resid_post_direct_ablated)
            model_results.logits[("indirect", "frozen", ablation_type)][layer, head] = (resid_post_indirect_ablated / model_results.scale) @ model.W_U + model.b_U
            model_results.logits[("indirect", "unfrozen", ablation_type)][layer, head] = (resid_post_indirect_ablated / resid_post_indirect_ablated.std(-1, keepdim=True)) @ model.W_U + model.b_U

            # Do exactly the same thing, just with a different (fancier!) form of ablation which preserves the pure copy-suppression mechanism.
            resid_post_indirect_ablated_CS = resid_post_orig + (resid_post_both_ablated_CS - resid_post_direct_ablated_CS)
            model_results.logits[("indirect", "frozen", ablation_type, "CS")][layer, head] = (resid_post_indirect_ablated_CS / model_results.scale) @ model.W_U + model.b_U
            model_results.logits[("indirect", "unfrozen", ablation_type, "CS")][layer, head] = (resid_post_indirect_ablated_CS / resid_post_indirect_ablated_CS.std(-1, keepdim=True)) @ model.W_U + model.b_U

            # ! Calculate new loss (and while doing this, get all the tensors which tell us if this example is copy suppression)

            # For each of these six logits, calculate loss differences (for regular ablation, and CS-preserving ablation)
            for effect in ["both", "direct", "indirect"]:
                for ln_mode in ["frozen", "unfrozen"]:
                    # Get the logits, compute and store the corresponding loss
                    logits = model_results.logits[(effect, ln_mode, ablation_type)][layer, head]
                    loss_ablated_minus_orig = model.loss_fn(logits, toks, per_token=True) - model_results.loss_orig
                    model_results.loss_diffs[(effect, ln_mode, ablation_type)][layer, head] = loss_ablated_minus_orig
                    # Get the logits for CS-preserving ablation, compute and store the corresponding loss
                    logits_CS = model_results.logits[(effect, ln_mode, ablation_type, "CS")][layer, head]
                    loss_CS_minus_orig = model.loss_fn(logits_CS, toks, per_token=True) - model_results.loss_orig
                    model_results.loss_diffs[(effect, ln_mode, ablation_type, "CS")][layer, head] = loss_CS_minus_orig

                    # ! Calculate the tensor which tells us which examples are copy suppression

                    # Define a tensor to store the results
                    batch_size, seq_len = toks.shape
                    is_copy_suppression = t.zeros((batch_size, seq_len-1), dtype=t.int32)

                    # Get the ratio in loss differences
                    loss_ablated_minus_CS = loss_ablated_minus_orig - loss_CS_minus_orig
                    loss_ratio = loss_ablated_minus_CS / loss_ablated_minus_orig

                    # Figure out which ones are copy suppression, according to our threshold
                    is_copy_suppression[loss_ratio > loss_ratio_threshold] = 1

                    # Filter "inconclusive" values, i.e. ones which don't have enough source tokens
                    # (we apply the percentile filter after we actually get the results)
                    
                    non_enough_src_toks = (unembedding_components_top10.values[:, :-1] > -1e4).int().sum(dim=-1) < num_top_src_tokens
                    is_copy_suppression = t.where(non_enough_src_toks, 2, is_copy_suppression)

                    # Store the results
                    # ! Temporarily storing more than just the core `is_copy_suppression` results, so I can bug-fix
                    model_results.is_copy_suppression[(effect, ln_mode, ablation_type)][layer, head] = {
                        "CS": is_copy_suppression,
                        "LR": loss_ratio,
                        "L_ORIG": model_results.loss_orig,
                        "L_CS": loss_CS_minus_orig + model_results.loss_orig,
                        "L_ABL": loss_ablated_minus_orig + model_results.loss_orig,
                    }

    if verbose: print("Finishing...")

    model_results.clear()

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

    return model_results