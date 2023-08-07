# This file contains functions to get results from my model. I use them to generate the visualisations found on the Streamlit pages.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
from typing import Dict, Any, Tuple, List, Optional, Literal
from transformer_lens import HookedTransformer, utils
from functools import partial
import einops
from dataclasses import dataclass
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int, Float, Bool
import torch as t
from collections import defaultdict
import time
from torch import Tensor
import pandas as pd
import numpy as np
from copy import copy
import gzip
import pickle

from transformer_lens.rs.callum2.cspa.cspa_semantic_similarity import (
    concat_lists,
    make_list_correct_length,
    get_list_with_no_repetitions,
)
from transformer_lens.rs.callum2.generate_st_html.utils import (
    ST_HTML_PATH,
)

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
    ' at', ' of', 'to', ' now', "'s", 'The', ".", ",", "?", "!", " '", "'", ' "', '"', 'you', 'your', ' you', ' your', ' once', ' and', ' all', 'Now', ' He',
    ' her', ' my', ' your',
    '<|endoftext|>',
    # Need to go through these words and add more to them, I suspect this list is minimal
]


def first_occurrence(array_1D):
    series = pd.Series(array_1D)
    duplicates = series.duplicated(keep='first')
    inverted = ~duplicates
    return inverted.values

def first_occurrence_2d(tensor_2D):
    device = tensor_2D.device
    array_2D = utils.to_numpy(tensor_2D)
    return t.from_numpy(np.array([first_occurrence(row) for row in array_2D])).to(device)



def gram_schmidt(vectors: Float[Tensor, "... d num"]) -> Float[Tensor, "... d num"]:
    '''
    Performs Gram-Schmidt orthonormalization on a batch of vectors, returning a basis.

    `vectors` is a batch of vectors. If it was 2D, then it would be `num` vectors each with length
    `d`, and I'd want a basis for these vectors in d-dimensional space. If it has more dimensions 
    at the start, then I want to do the same thing for all of them (i.e. get multiple independent
    bases).

    If the vectors aren't linearly independent, then some of the basis vectors will be zero (this is
    so we can still batch our projections, even if the subspace rank for each individual projection
    is not equal.
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
        
        # Normalize this vector (we can set it to zero if it's not linearly independent)
        basis_vec_norm = basis[..., i].norm(dim=-1, keepdim=True)
        basis[..., i] = t.where(
            basis_vec_norm > 1e-4,
            basis[..., i] / basis_vec_norm,
            t.zeros_like(basis[..., i])
        )
    
    return basis



def project(
    vectors: Float[Tensor, "... d"],
    proj_directions: Float[Tensor, "... d num"],
    only_keep: Optional[Literal["pos", "neg"]] = None,
    gs: bool = True,
    return_coeffs: bool = False,
):
    '''
    `vectors` is a batch of vectors, with last dimension `d` and all earlier dimensions as batch dims.

    `proj_directions` is either the same shape as `vectors`, or has an extra dim at the end.

    If they have the same shape, we project each vector in `vectors` onto the corresponding direction
    in `proj_directions`. If `proj_directions` has an extra dim, then the last dimension is another 
    batch dim, i.e. we're projecting each vector onto a subspace rather than a single vector.
    '''
    # If we're only projecting onto one direction, add a dim at the end (for consistency)
    if proj_directions.shape == vectors.shape:
        proj_directions = proj_directions.unsqueeze(-1)
    # Check shapes
    assert proj_directions.shape[:-1] == vectors.shape
    assert not((proj_directions.shape[-1] > 20) and gs), "Shouldn't do too many vectors, GS orth might be computationally heavy I think"

    # We might want to have done G-S orthonormalization first
    proj_directions_basis = gram_schmidt(proj_directions) if gs else proj_directions

    components_in_proj_dir = einops.einsum(
        vectors, proj_directions_basis,
        "... d, ... d num -> ... num"
    )
    if return_coeffs: return components_in_proj_dir
    
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
    layer: int,
    return_type: Literal["logits", "resid_post", "loss"] = "logits",
    toks: Optional[Int[Tensor, "batch seq"]] = None,
) -> Float[Tensor, "batch seq d_vocab"]:
    '''
    Performs a forward pass starting from an intermediate point.

    For instance, if layer=10, this will apply the TransformerBlocks from
    layers 10, 11 respectively, then ln_final and unembed.

    Also, if return_resid_post = True, then it just returns the final value
    of the residual stream, i.e. omitting ln_final and unembed.
    '''
    assert return_type in ["logits", "resid_post", "loss"]

    resid = resid_pre
    for i in range(layer, model.cfg.n_layers):
        resid = model.blocks[i](resid)
    
    if (return_type == "resid_post"): return resid

    resid_scaled = model.ln_final(resid)
    logits = model.unembed(resid_scaled)

    if return_type == "logits":
        return logits
    elif return_type == "resid_post":
        return resid
    elif return_type == "loss":
        assert toks is not None
        return model.loss_fn(logits, toks, per_token=True)
    


def get_effective_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:

    # TODO - make this consistent (i.e. change the func in `generate_bag_of_words_quad_plot` to also return W_U and W_E separately)

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

    resid_pre = W_E.unsqueeze(0)
    pre_attention = model.blocks[0].ln1(resid_pre)
    attn_out = einops.einsum(
        pre_attention, 
        model.W_V[0],
        model.W_O[0],
        "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
    )
    resid_mid = attn_out + resid_pre
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    t.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_U": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE
    }




def get_cspa_results(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_head: Tuple[int, int],
    components_to_project: List[str],
    K_unembeddings: Optional[int] = None,
    K_semantic: int = 10,
    semantic_dict: dict = {},
    effective_embedding: str = "W_E (including MLPs)",
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    use_cuda: bool = False,
    return_dla: bool = False,
) -> Tuple[
        Dict[str, Float[Tensor, "batch seq-1"]],
        Dict[Tuple[str, str], List[Tuple[str, str]]],
    ]:
    '''
    Short explpanation of the copy-suppression preserving ablation (hereby CSPA), with
    the following arguments:

        components_to_project = ["o"]
        K_semantic = 3
        K_unembeddings = 5

        For every souce token s, we pick the 3 tokens which it's most semantically
        related to - call this set S*.

        For every destination token d, we look at the union of all semantically similar
        tokens S* for the source tokens before it in context, and pick the 5 which
        are most predicted (i.e. the highest logit lens). This gives us 5 pairs (s, s*).

        For each source token s which is in one of these pairs, we take the result vector
        which is moved from s -> d, and project it onto the unembeddings of all s* which
        it appears in a pair with. If a source token doesn't appear in one of these pairs,
        we mean ablate that vector.

            Summary: information is moved from s -> d if and only if s is semantically similar 
            to some token s* which is being predicted at d, and in this case the information
            which is moved is restricted to the subspace of the unembedding of s*.

    
    A few notes / explanations:

        > There are options for "q" and "v" to be iin the components to project, but this
          currently isn't how the best version of this ablation method works.
        > result_mean is the vector we'll use for ablation, if supplied. It'll map e.g. 
          (10, 7) to the mean result vector for each seqpos (hopefully seqpos is larger than 
          that for toks).

    Return type:

        > A dictionary of the results, with "loss", "loss_ablated", and "loss_projected"
        > A dict mapping (batch_idx, d) to the list of (s, s*) which we preserve
          in our ablation.
    '''

    # ====================================================================
    # STEP 0: Setup
    # ====================================================================

    W_EE_dict = get_effective_embedding(model)
    W_EE = W_EE_dict[effective_embedding]
    W_EE = W_EE / W_EE.std(dim=-1, keepdim=True)
    W_U = model.W_U

    batch_size, seq_len = toks.shape
    
    FUNCTION_TOKS = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze()
    
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()

    current_device = str(next(iter(model.parameters())).device)
    if use_cuda and current_device == "cpu":
        model = model.cuda()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cpu()



    # ====================================================================
    # STEP 1: Get effective embeddings, and semantically similar unembeddings
    # ====================================================================

    effective_embeddings = W_EE[toks] # [batch seqK d_model]
    if K_semantic > 1:
        # Flatten all tokens, and convert them into str toks
        str_toks_flat = model.to_str_tokens(toks.flatten(), prepend_bos=False)
        # Use our dictionary to find the top `K_semantic` semantically similar tokens for each one
        semantically_similar_str_toks = concat_lists([make_list_correct_length(concat_lists(semantic_dict[str_tok]), K_semantic, pad_tok='======') for str_tok in str_toks_flat])
        # Turn these into tokens
        semantically_similar_toks = model.to_tokens(semantically_similar_str_toks, prepend_bos=False)
        semantically_similar_toks = semantically_similar_toks.reshape(batch_size, seq_len, K_semantic) # [batch seqK K_semantic]
    else:
        semantically_similar_toks = toks.unsqueeze(-1) # [batch seqK 1]

    # Get these tokens' unembeddings
    unembeddings = W_U.T[toks] # [batch seqK d_model]



    # ====================================================================
    # STEP 2: Define hook functions to project or ablate
    # ====================================================================
    
    # Get all hook names
    # TODO - change names to "pattern_hook_name" and "pattern_hook_fn" etc
    hook_name_resid = utils.get_act_name("resid_pre", 10)
    hook_name_resid_final = utils.get_act_name("resid_post", 11)
    hook_name_q_input = utils.get_act_name("q_input", 10)
    hook_name_k_input = utils.get_act_name("k_input", 10)
    hook_name_v_input = utils.get_act_name("v_input", 10)
    hook_name_result = utils.get_act_name("result", 10)
    hook_name_v = utils.get_act_name("v", 10)
    hook_name_pattern = utils.get_act_name("pattern", 10)
    hook_name_scale = utils.get_act_name("scale", 10, "ln1")
    hook_name_scale_final = utils.get_act_name("scale")
    hook_names_to_cache = [hook_name_k_input, hook_name_q_input, hook_name_v_input, hook_name_scale, hook_name_scale_final, hook_name_resid, hook_name_resid_final, hook_name_result]
    


    # ==================================
    # STEP 2A: scales & storage
    # ==================================
    
    # We need to freeze scale factors. This is messy, cause keys/queries/values resp. The
    # way I'm getting around this is by using the fact that adding this hook will add it 3 times,
    # and it'll refer to a hook-ctx `index` variable, only activating when the index is right.
    def hook_fn_freeze_scale(
        scale: Float[Tensor, "batch seqK head 1"],
        hook: HookPoint,
        frozen_scale: Float[Tensor, "batch seqK 1"],
        component: Literal["q", "k", "v"],
    ):
        # First time this hook is called, index will be 0. Second time, it'll be 1, then 2, then deleted.
        if "index" not in hook.ctx:
            hook.ctx["index"] = 0
        # If current value of index matches the component we want to freeze layernorm at, then freeze.
        component_index = {"q": 0, "k": 1, "v": 2}[component]
        if component_index == hook.ctx["index"]:
            scale[:, :, 7] = frozen_scale
        # Increment index, and delete if we've incremented 3 times.
        hook.ctx["index"] += 1
        if hook.ctx["index"] == 3:
            del hook.ctx["index"]

        return scale

    def hook_fn_store_v_or_attn_in_ctx(activation: Tensor, hook: HookPoint):
        if hook.name.endswith("hook_pattern"):
            hook.ctx["pattern"] = activation[:, 7] # [batch seqQ seqK]
        elif hook.name.endswith("hook_v"):
            hook.ctx["v"] = activation[:, :, 7] # [batch seqK d_head]
    


    # ==================================
    # STEP 2B: v
    # ==================================

    def hook_fn_project_v(
        v_input: Float[Tensor, "batch seqK head d_model"],
        hook: HookPoint,
    ):
        '''
        Projects the value input onto the effective embedding for the source token.
        '''
        v_input_head = v_input[:, :, 7]
        v_input_head_mean = einops.reduce(v_input_head, "batch seqK d_model -> seqK d_model")

        v_input[:, :, 7] = project(
            vectors = v_input_head - v_input_head_mean,
            proj_directions = effective_embeddings,
        ) + v_input_head_mean
        
        return v_input



    # ==================================
    # STEP 2C: o (i.e. results)
    # This is hard, because we need to actually recompute it starting from v and attn.
    # ==================================

    def hook_fn_result(
        result: Float[Tensor, "batch seqQ head d_model"],
        hook: HookPoint,
        resid_pre: Float[Tensor, "batch seqQ d_model"],
        final_scale: Optional[Float[Tensor, "batch seqQ 1"]] = None
    ):
        '''
        Projects each result vector onto the unembeddings for semantically similar tokens.

        We assume that `v` and `pattern` for this head (which might have been changed from their 
        clean values) have been stored in their respective hook contexts.
        '''
        # Get v and pattern from previous hook.ctx
        v: Tensor = model.hook_dict[hook_name_v].ctx.pop("v") # [batch seqK d_head]
        pattern: Tensor = model.hook_dict[hook_name_pattern].ctx.pop("pattern") # [batch seqQ seqK]            
        
        # Multiply by output matrix, then by attention probabilities
        output = v @ model.W_O[10, 7] # [batch seqK d_model]
        output_attn = einops.einsum(output, pattern, "batch seqK d_model, batch seqQ seqK -> batch seqQ seqK d_model")
        output_attn_mean_ablated = einops.reduce(output_attn, "batch seqQ seqK d_model -> seqQ 1 d_model", "mean")
        # We want to use the results supplied for mean ablation, if we're short on data here
        if batch_size * seq_len < 1000:
            assert result_mean is not None, "You should be using an externally supplied mean ablation vector for such a small dataset."
            output_attn_pre_mean_ablation = einops.einsum(
                result_mean[(10, 7)][:seq_len], pattern,
                "seqQ d_model, batch seqQ seqK -> batch seqQ seqK d_model"
            )
            output_attn_mean_ablated = einops.reduce(output_attn_pre_mean_ablation, "batch seqQ seqK d_model -> seqQ 1 d_model", "mean")

        # Get the unembeddings we'll be projecting onto (also get the dict of (s, s*) pairs and store in context)
        semantically_similar_unembeddings, s_sstar_pairs = get_top_predicted_semantically_similar_tokens(
            toks=toks,
            resid_pre=resid_pre,
            semantically_similar_toks=semantically_similar_toks,
            K_unembeddings=K_unembeddings,
            function_toks=FUNCTION_TOKS,
            model=model,
            final_scale=final_scale,
        )
        hook.ctx["s_sstar_pairs"] = s_sstar_pairs

        # Perform the projection onto semantically similar tokens (make sure to do it wrt the mean)
        output_attn_projected = project(
            vectors = output_attn - output_attn_mean_ablated,
            proj_directions = semantically_similar_unembeddings,
        ) + output_attn_mean_ablated

        # Sum over key-side vectors to get new head result
        # ? (don't override the BOS token attention, because it's more appropriate to preserve this information I think)
        # output_attn_projected[:, :, 0, :] = output_attn[:, :, 0, :]
        head_result = einops.reduce(output_attn_projected, "batch seqQ seqK d_model -> batch seqQ d_model", "sum")

        result[:, :, 7] = head_result
        
        return result

    def hook_fn_cache_result(
        result: Float[Tensor, "batch seqQ head d_model"],
        hook: HookPoint,
    ):
        hook.ctx["result"] = result[:, :, 7]


    # ==================================
    # STEP 2D: q
    # This is hard, because we need to actually recompute it from the resid pre.
    # See archived code at the end of this doc.
    # ==================================

    def hook_fn_cache_scores(
        scores: Float[Tensor, "batch head seqQ seqK"],
        hook: HookPoint,
    ):
        hook.ctx["scores"] = scores[:, 7]


    # ====================================================================
    # STEP -1: Return the results
    # ====================================================================

    # First, get clean results (also use this to get residual stream before layer 10)
    model.reset_hooks()
    logits, cache = model.run_with_cache(
        toks,
        return_type = "logits",
        names_filter = lambda name: name in hook_names_to_cache
    )
    loss = model.loss_fn(logits, toks, per_token=True)
    resid_post_final = cache["resid_post", -1] # [batch seqQ d_model]
    resid_pre = cache["resid_pre", 10] # [batch seqK d_model]
    # * Weird error which I should fix; sometimes it seems like scale isn't recorded as different across heads
    scale = cache["scale", 10, "ln1"]
    if scale.ndim == 4:
        scale = cache["scale", 10, "ln1"][:, :, 7] # [batch seqK 1]
    head_result_orig = cache["result", 10][:, :, 7] # [batch seqQ d_model]
    final_scale = cache["scale"] # [batch seq 1]
    del cache
    
    # Secondly, perform complete ablation (via a direct calculation)
    head_result_orig_mean_ablated = einops.reduce(head_result_orig, "batch seqQ d_model -> seqQ d_model", "mean")
    if batch_size * seq_len < 1000:
        assert result_mean is not None, "You should be using an externally supplied mean ablation vector for such a small dataset."
        head_result_orig_mean_ablated = result_mean[(10, 7)][:seq_len]
    resid_post_final_mean_ablated = resid_post_final + (head_result_orig_mean_ablated - head_result_orig) # [batch seq d_model]
    logits_mean_ablated = (resid_post_final_mean_ablated / final_scale) @ model.W_U + model.b_U
    loss_mean_ablated = model.loss_fn(logits_mean_ablated, toks, per_token=True)
    model.reset_hooks()

    # Thirdly, get results after projecting
    # Add hook functions to project the decomposed result vectors along the semnatically similar unembeddings
    if "o" in components_to_project:
        model.add_hook(hook_name_v, hook_fn_store_v_or_attn_in_ctx)
        model.add_hook(hook_name_pattern, hook_fn_store_v_or_attn_in_ctx)
        model.add_hook(hook_name_result, partial(hook_fn_result, resid_pre=resid_pre, final_scale=final_scale))
    # Add hooks to project the value input along the source tokens' effective embeddings
    if "v" in components_to_project:
        model.add_hook(hook_name_v_input, hook_fn_project_v)
        model.add_hook(hook_name_scale, partial(hook_fn_freeze_scale, frozen_scale=scale, component="v"))
    # Add hooks to project the query input along each source tokens' semantically similar unembedding
    if "q" in components_to_project:
        pass
        # model.add_hook(hook_name_pattern, partial(hook_fn_project_q_via_pattern, resid_pre=resid_pre))
        # model.add_hook(utils.get_act_name("attn_scores", 10), partial(hook_fn_cache_scores))
    # Cache the result when this is all done, and then run block 10 just so we can get this new result
    model.add_hook(hook_name_result, hook_fn_cache_result)
    _ = model.blocks[10](resid_pre)
    s_sstar_pairs = model.hook_dict[hook_name_result].ctx.pop("s_sstar_pairs", None)
    # Take this new result vector
    head_result_cspa = model.hook_dict[hook_name_result].ctx.pop("result") # [batch seq d_model]
    resid_post_final_cspa = resid_post_final + (head_result_cspa - head_result_orig) # [batch seq d_model]
    logits_cspa = (resid_post_final_cspa / final_scale) @ model.W_U + model.b_U
    loss_cspa = model.loss_fn(logits_cspa, toks, per_token=True)
    model.reset_hooks(clear_contexts=False)

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

    cspa_results = {
        "loss": loss,
        "loss_projected": loss_cspa,
        "loss_ablated": loss_mean_ablated,
    }
    if return_dla:
        cspa_results["dla"] = ((head_result_cspa - head_result_orig_mean_ablated) / final_scale) @ model.W_U
        cspa_results["logits"] = logits_cspa
        cspa_results["logits_orig"] = logits

    return cspa_results, s_sstar_pairs



def get_cspa_results_batched(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    max_batch_size: int,
    negative_head: Tuple[int, int],
    components_to_project: List[str],
    K_unembeddings: Optional[int] = None,
    K_semantic: int = 10,
    semantic_dict: dict = {},
    effective_embedding: str = "W_E (including MLPs)",
    use_cuda: bool = False,
    verbose: bool = False,
    return_dla: bool = False,
    start_idx: int = 0,
) -> Dict[str, Float[Tensor, "batch seq-1"]]:
    '''
    Gets results from CSPA, by splitting the tokens along batch dimension and running it several 
    times. This allows me to use batch sizes of 1000+ without getting CUDA errors.
    '''
    cspa_results_list = []
    chunks = toks.shape[0] // max_batch_size
    lower_upper_list = []

    for i, _toks in enumerate(t.chunk(toks, chunks=chunks)):

        # Set the values of `lower` and `upper`, i.e. batch sizes (useful for saving files)
        lower, upper = i*max_batch_size, (i+1)*max_batch_size
        lower_upper_list.append((lower, upper))
        # We only start saving results from `start_idx` onwards (so we don't waste what we've already saved)
        if lower < start_idx: continue
        if verbose: print(f"Running batch {i+1}/{chunks} of size {_toks.shape[0]} ... ", end="\r"); t0 = time.time()
        
        # Get new results
        result_mean = None
        cspa_results, s_sstar_pairs = get_cspa_results(
            model,
            _toks,
            negative_head,
            components_to_project,
            K_unembeddings,
            K_semantic,
            semantic_dict,
            effective_embedding,
            result_mean,
            use_cuda,
            return_dla,
        )

        # Save results, and clear cache
        with open(ST_HTML_PATH / f"_CSPA_and_S_SSTAR_{lower}_{upper}.pkl", "wb") as f:
            pickle.dump((cspa_results, s_sstar_pairs), f)
        del cspa_results, s_sstar_pairs
        t.cuda.empty_cache()
        if verbose: print(f"Running batch {i+1}/{chunks} of size {_toks.shape[0]} ... {time.time()-t0:.2f}")

    # Load all results from where they are saved
    if verbose:
        print("\nLoading all results from where they were saved ...", end="\r")
        t0 = time.time()
    for (lower, upper) in lower_upper_list:
        _cspa_results, _s_sstar_pairs = pickle.load(open(ST_HTML_PATH / f"_CSPA_and_S_SSTAR_{lower}_{upper}.pkl", "rb"))
        if lower == 0:
            # This is the first batch, so just set the results to be the thing you read
            s_sstar_pairs = _s_sstar_pairs
            cspa_results = {k: v.cpu() for k, v in _cspa_results.items()}
        else:
            # Concatenate the results tensors (for loss, ablated loss, and CSPA (projected) loss)
            for key in cspa_results:
                cspa_results[key] = t.concat([cspa_results[key], _cspa_results[key].cpu()], dim=0)
            # Add the (s, s*) pairs, making sure to increment by the appropriate batch index
            for (b, sQ), v in _s_sstar_pairs.items():
                s_sstar_pairs[(b+upper, sQ)] = v

    if verbose:
        print(f"Loading all results from where they were saved ... {time.time()-t0:.2f}")
        print("Deleting HTML plots we no longer need...")
    for (lower, upper) in lower_upper_list:
        os.remove(ST_HTML_PATH / f"_CSPA_and_S_SSTAR_{lower}_{upper}.pkl")

    return cspa_results, s_sstar_pairs




def get_top_predicted_semantically_similar_tokens(
    toks: Int[Tensor, "batch seq"],
    resid_pre: Float[Tensor, "batch seqK d_model"],
    semantically_similar_toks: Int[Tensor, "batch seq K_semantic"],
    K_unembeddings: int,
    function_toks: Int[Tensor, "tok"],
    model: HookedTransformer,
    final_scale: Optional[Float[Tensor, "batch seqQ 1"]] = None,
    s_sstar_contains_indices: bool = False,
):
    '''
    Arguments:

        toks: [batch seq]
            The source tokens.
        
        resid_pre: [batch seqK d_model]
            The residual stream before the head, which we'll use to figure out which tokens in `toks`
            are predicted with highest probability.

        semantically_similar_toks: [batch seqK K_semantic]
            Semantically similar tokens for each source token.
        
        function_toks: [tok]
            1D array of function tokens (we filter for pairs (s, s*) where s is not a function token).

    Returns:

        semantically_similar_unembeddings: [batch seqQ seqK d_model K_unembeddings]
            The unembeddings of the semantically similar tokens, with all the vectors except the
            ones we're actually using set to zero.
        
        mask: [batch seqQ seqK K_semantic]
            The causal mask which we'll be applying once we project the unembeddings.

        s_sstar_pairs: defaultdict(list)
            Keys are (b, sQ) indices, values are lists of (s, s*) str_tok pairs which are preserved
            in CSPA, where s is a source token and s* is a semantically similar token that is predicted
            at the destination position.
    '''
    semantically_similar_unembeddings = model.W_U.T[semantically_similar_toks].transpose(-1, -2) # [batch seqK d_model K_semantic]
    batch_size, seq_len = toks.shape
    s_sstar_pairs = defaultdict(list)
    
    # If K_unembeddings is None, then we have no restrictions, and we use all the top K_semantic toks for each seqK
    if K_unembeddings is None:
        semantically_similar_unembeddings = einops.repeat(
            semantically_similar_unembeddings,
            "batch seqK d_model K_semantic -> batch seqQ seqK d_model K_semantic",
            seqQ = seq_len,
        )

    # Else, we filter down the sem similar unembeddings to only those that are predicted
    else:
        # Get logit lens from current value of residual stream
        # logit_lens[b, sQ, sK, K_s] = prediction at destination token (b, sQ), for the K_s'th semantically similar token to source token (b, sK)
        resid_pre_scaled = resid_pre if (final_scale is None) else resid_pre / final_scale
        logit_lens = einops.einsum(
            resid_pre_scaled, semantically_similar_unembeddings,
            "batch seqQ d_model, batch seqK d_model K_semantic -> batch seqQ seqK K_semantic",
        )

        # MASK: make sure function words are never the source token (because we've observed that the QK circuit has managed to learn this)
        is_fn_word = (toks[:, :, None] == function_toks).any(dim=-1) # [batch seqK]
        logit_lens = t.where(einops.repeat(is_fn_word, "batch seqK -> batch 1 seqK 1"), -1e9, logit_lens)
        # MASK: apply causal mask
        seqQ_idx = einops.repeat(t.arange(seq_len), "seqQ -> 1 seqQ 1 1").to(logit_lens.device)
        seqK_idx = einops.repeat(t.arange(seq_len), "seqK -> 1 1 seqK 1").to(logit_lens.device)
        logit_lens = t.where(seqQ_idx < seqK_idx, -1e9, logit_lens)
        # MASK: each source token should only be counted at its first instance
        # Note, we apply this mask to get our topk values (so no repetitions), but we don't want to apply it when we're choosing pairs to keep
        first_occurrence_mask = einops.repeat(first_occurrence_2d(toks), "batch seqK -> batch 1 seqK 1")
        logit_lens_for_topk = t.where(~first_occurrence_mask, -1e9, logit_lens)

        # Get the top predicted src-semantic-neighbours s* for each destination token d
        top_K_and_Ksem_per_dest_token_values = t.topk(logit_lens_for_topk.flatten(-2, -1), K_unembeddings, dim=-1).values[..., [[-1]]] # [batch seqQ 1 1]
        # We also want to get the list of (s, s*) for analysis afterwards
        top_K_and_Ksem_mask = (logit_lens + 1e-6 >= top_K_and_Ksem_per_dest_token_values) # [batch seqQ seqK K_s]
        top_K_and_Ksem_per_dest_token = t.nonzero(top_K_and_Ksem_mask) # [n batch_seqQ_seqK_K_s], n >= batch * seqQ * K_u (more if we're double-counting source tokens)
        for b, sQ, sK, K_s in top_K_and_Ksem_per_dest_token.tolist():
            s = (f"[{sK}] " if s_sstar_contains_indices else "") + repr(model.to_single_str_token(toks[b, sK].item()))
            sstar = repr(model.to_single_str_token(semantically_similar_toks[b, sK, K_s].item()))
            LL = logit_lens[b, sQ, sK, K_s].item() # for sorting with (and maybe I'll add it to the visualization too)
            # Make sure we only count each (s, s*) pair once (since this dict is for the HTML visualisations)
            if (s, sstar) not in list(map(lambda x: x[1:], s_sstar_pairs[(b, sQ)])):
                s_sstar_pairs[(b, sQ)].append((LL, s, sstar))
        # Make sure we rank order the entries in each dictionary by how much they're being predicted
        for (b, sQ), s_star_list in s_sstar_pairs.items():
            s_sstar_pairs[(b, sQ)] = sorted(s_star_list, key = lambda x: x[0], reverse=True)

        # Use this boolean mask to set some of the unembedding vectors to zero
        unembeddings = einops.repeat(semantically_similar_unembeddings, "batch seqK d_model K_semantic -> batch 1 seqK d_model K_semantic")
        mask = einops.repeat(top_K_and_Ksem_mask, "batch seqQ seqK K_semantic -> batch seqQ seqK 1 K_semantic")
        semantically_similar_unembeddings = unembeddings * mask.float()

    return semantically_similar_unembeddings, s_sstar_pairs



# ? This code doesn't work super well; projecting queries is meh.
# def hook_fn_project_q_via_pattern(
#     pattern: Float[Tensor, "batch head seqQ seqK"],
#     hook: HookPoint,
#     resid_pre: Float[Tensor, "batch seqK d_model"],
# ):
#     '''
#     Projects the query input onto the effective embedding for the source token.
#     '''
#     # Apply LN scale
#     scale = resid_pre.std(dim=-1, keepdim=True)
#     # Get the thing we'll be projecting along the sematic unembeddings for each src token
#     query_input = einops.repeat(
#         resid_pre,
#         "batch seqQ d_model -> batch seqQ seqK d_model",
#         seqK = seq_len,
#     )
#     query_input_mean = query_input.mean(dim=0, keepdim=True)
#     semantically_similar_unembeddings_repeated = einops.repeat(
#         semantically_similar_unembeddings,
#         "batch seqK d_model K_semantic -> batch seqQ seqK d_model K_semantic",
#         seqQ = seq_len,
#     )
#     # Project onto the appropriate semantically similar tokens
#     query_input_projected = project(
#         vectors = query_input - query_input_mean,
#         proj_directions = semantically_similar_unembeddings_repeated,
#     ) # [batch seqQ seqK d_model] #  + query_input_mean
#     # Rescale by norm of source tokens (this is how we control for function words!)
#     # TODO - this is definitely not the optimal way to do it, not sure what is optimal yet
#     query_input_projected = query_input_projected * einops.repeat(
#         unembeddings.norm(dim=-1), #  / unembeddings.norm(dim=-1).max(),
#         "batch seqK -> batch seqQ seqK d_model",
#         seqQ = seq_len,
#         d_model = query_input_projected.shape[-1],
#     )
#     # Add mean back in
#     query_input_projected = query_input_projected + query_input_mean

#     # Now we can get keys and queries, and calculate our new attention scores (scaled and masked!)
#     keys = (resid_pre / scale) @ model.W_K[10, 7] + model.b_K[10, 7]
#     queries = (query_input_projected / scale.unsqueeze(-1)) @ model.W_Q[10, 7] + model.b_Q[10, 7]
#     new_attn_scores = einops.einsum(
#         queries, keys,
#         "batch seqQ seqK d_head, batch seqK d_head -> batch seqQ seqK",
#     ) / (model.cfg.d_head ** 0.5)
#     new_attn_scores.masked_fill_(t.triu(t.ones_like(new_attn_scores), diagonal=1).bool(), -float("inf"))
#     new_pattern = new_attn_scores.softmax(dim=-1)

#     # We want to make sure that attention prob to the zeroth token is same as before (as a baseline)
#     # e.g. if original attn to 0 was very high, we'll be scaling down the new not-to-0 attn probs
#     new_pattern[:, 1:, 1:] *= (1 - pattern[:, 7, 1:, [0]]) / (1 - new_pattern[:, 1:, [0]])
#     new_pattern[:, :, 0] = pattern[:, 7, :, 0]
#     # t.testing.assert_close(new_pattern.sum(dim=-1), t.ones_like(new_pattern.sum(dim=-1)))

#     hook.ctx["info"] = (pattern[:, 7].clone(), new_attn_scores.clone(), new_pattern.clone())

#     pattern[:, 7] = new_pattern
#     return pattern