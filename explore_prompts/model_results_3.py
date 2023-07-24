# This file contains functions to get results from my model. I use them to generate the visualisations found on the Streamlit pages.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
for root_dir in [
    os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts", # For Arthur's branch
    "/app/seri-mats-2023-streamlit-pages/explore_prompts", # For Streamlit page (public)
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/explore_prompts", # For Arthur's branch
]:
    if os.path.exists(root_dir):
        break
os.chdir(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

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
    data: Dict[Head, Tensor]
    '''
    Stores results for each head. This is just a standard dictionary, but you can index it like [layer, head]
    (this is just a convenience thing, so I don't have to wrap the head in a tuple).
    '''
    def __init__(self, data=None):
        self.data = data or {}
    def __getitem__(self, layer_and_head: Head):
        return self.data[layer_and_head]
    def __setitem__(self, layer_and_head: Head, value):
        self.data[layer_and_head] = value
    def keys(self):
        return self.data.keys() # repetitive code, should improve
    def items(self):
        return self.data.items()


class LayerResults:
    data: Dict[int, Any]
    '''
    Similar syntax to HeadResults, but the keys are ints (layers) not (layer, head) tuples.
    '''
    def __init__(self, data=None):
        self.data = data or {}
    def __getitem__(self, layer: int):
        return self.data[layer]
    def __setitem__(self, layer: int, value):
        self.data[layer] = value
    def keys(self):
        return self.data.keys() # repetitive code, should improve
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
    def keys(self):
        return self.data.keys() # repetitive code, should improve
    def items(self):
        return self.data.items()


class ModelResults:
    '''
    All the datatypes are one of 3 things:

    1. Just plain old tensors

    2. HeadResults: this is nominally a dictionary mapping (layer, head) tuples to values,
       the only difference is you can index it with [layer, head], like you can for tensors.
       Also similarly there's LayerResults.

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
        self.z: HeadResults = HeadResults()

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

        # This is the "prediction-attention" part, and it's how we perform copy-suppression preserving ablation
        self.logit_lens: LayerResults = LayerResults()
        self.E_sq: HeadResults = HeadResults()
        self.unembedding_projection_dirs: HeadResults = HeadResults()

        # This is to store random crap!
        self.misc: dict = {}


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

    # We might want to have done G-S orthonormalization first
    proj_directions_basis = gram_schmidt(proj_directions) if gs else proj_directions

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
    layer: int,
    return_type: Literal["logits", "resid_post", "both"] = "logits",
) -> Float[Tensor, "batch seq d_vocab"]:
    '''
    Performs a forward pass starting from an intermediate point.

    For instance, if layer=10, this will apply the TransformerBlocks from
    layers 10, 11 respectively, then ln_final and unembed.

    Also, if return_resid_post = True, then it just returns the final value
    of the residual stream, i.e. omitting ln_final and unembed.
    '''
    assert return_type in ["logits", "resid_post", "both"]

    resid = resid_pre
    for i in range(layer, model.cfg.n_layers):
        resid = model.blocks[i](resid)
    
    if (return_type == "resid_post"): return resid

    resid_scaled = model.ln_final(resid)
    logits = model.unembed(resid_scaled)
    
    return logits if (return_type == "logits") else (resid, logits)


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


def get_model_results(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_heads: List[Tuple[int, int]],
    use_cuda: bool = False,
    verbose: bool = False,
    K_semantic: int = 10,
    K_unembed: int = 10,
    effective_embedding: str = "W_E (including MLPs)",
    cspa: bool = True,
) -> ModelResults:
    '''
    Short explpanation of the copy-suppression preserving ablation (hereby CSPA):

        (1) For each destination token D, look at all the possible source tokens S. For each one, pick the
            top K_semantic tokens which attend to S via the full QK matrix. Call these pairs (S, Q).

                Key point is that this captures semantic similarity, rather than just pure copy-suppresion.
            
                Examples:
                    if S = " pier" then the Q's might be [" pier", " Pier", " piers", ...]
                    if S = "keley" then the Q's might be ["keley", " Berkeley", ...]

        (2) Find the top K_unembed pairs (sorted by component of W_U.T[Q] in the query-side residual stream).
        
                Examples:
                    if " Berkeley" is being predicted with high probability, and "keley" appears in context,
                    then the top (S, Q) pair will be ("keley", " Berkeley").

        (3) For each of these (S, Q) pairs, we DON'T ablate the component of attention moving from S -> D in
            the direction of the unembedding of Q. We ablate everything else.

    Why does this work? A simple version of copy-suppression would only project the vector from S onto the
    unembedding for S, but in practice we're suppressing context-based things not grammar-based things. For
    example, maybe the model believes some combination of "the next word has smth to do with piers" and "the
    next word doesn't have a capital letter". If 10.7 attends back to " Pier" then we want to suppress the 
    context-based signal, but we don't care about the grammar-based signal because the model has already dealt
    with this via a different mechanism. So we want to make sure our set of pairs (S, Q) contains both the pair
    (" Pier", " pier") and (" Pier", " Pier") - that way we correctly suppress the model's belief that the next
    token has something to do with the word pier (which is really the core of the copy-suppression mechanism).
    '''
    W_EE_dict = get_effective_embedding(model)
    W_EE = W_EE_dict[effective_embedding]
    W_EE = W_EE / W_EE.std(dim=-1, keepdim=True)
    W_U = model.W_U

    batch_size, seq_len = toks.shape
    
    # FUNCTION_TOKS = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze()
    
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()

    current_device = str(next(iter(model.parameters())).device)
    if use_cuda and current_device == "cpu":
        model = model.cuda()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cpu()

    model_results = ModelResults()

    if verbose:
        print(f"{'Running forward pass':<41} ... ", end="\r")
        t0 = time.time()

    # Cache the head results and attention patterns, and final ln scale, and residual stream pre heads
    # (note, this is preferable to using names_filter argument for cache, because you can't pick specific heads)

    def cache_head_result(result: Float[Tensor, "batch seq n_heads d_model"], hook: HookPoint, head: int):
        model_results.result[hook.layer(), head] = result[:, :, head]
    
    def cache_head_pattern(pattern: Float[Tensor, "batch n_heads seqQ seqK"], hook: HookPoint, head: int):
        model_results.pattern[hook.layer(), head] = pattern[:, head]
    
    def cache_head_v(v: Float[Tensor, "batch seq n_heads d_head"], hook: HookPoint, head: int):
        model_results.v[hook.layer(), head] = v[:, :, head]

    def cache_head_z(z: Float[Tensor, "batch seq n_heads d_head"], hook: HookPoint, head: int):
        model_results.z[hook.layer(), head] = z[:, :, head]
    
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
        model.add_hook(utils.get_act_name("z", layer), partial(cache_head_z, head=head))
        model.add_hook(utils.get_act_name("pattern", layer), partial(cache_head_pattern, head=head))
    # We also need some things at the very end of the model
    model.add_hook(utils.get_act_name("resid_post", model.cfg.n_layers-1), partial(cache_resid_post, key="orig"))
    model.add_hook(utils.get_act_name("scale"), cache_scale)

    # Run the forward pass, to cache all values (and get original logits)
    model_results.logits_orig, model_results.loss_orig = model(toks, return_type="both", loss_per_token=True)
    model.reset_hooks(clear_contexts=False)


    # ! Get all the tokens in context, and what unembeddings most attend to them, i.e. the pairs (S, Q)
    if verbose:
        print(f"{'Running forward pass':<41} ... {time.time()-t0:.2f}s")
        print(f"{'Computing logit lens (& CSPA)':<41} ... ", end="\r")
        t0 = time.time()

    for layer, head in negative_heads:

        # We get the logit lens, so that (for each head) we can figure out what to project onto
        resid_pre = model_results.resid_pre[layer]
        scale = model_results.scale_attn[layer]
        resid_pre_scaled: Float[Tensor, "batch seqQ d_model"] = resid_pre / scale
        assert t.all((resid_pre_scaled.norm(dim=-1) - model.cfg.d_model ** 0.5) < 1e-4)
        logit_lens: Float[Tensor, "batch seqQ d_vocab"] = resid_pre_scaled @ W_U
        model_results.logit_lens[layer] = logit_lens

        if cspa:
            # E(s,q) are the tokens s.t. W_EE[s] is negatively copied to -W_U[q]
            # i.e. E_sq[b, sK, :] are the token IDs of all such q, for the source token s at position sK
            W_EE_toks: Float[Tensor, "batch seqK d_model"] = W_EE[toks]
            # W_QK = model.W_Q[layer, head] @ model.W_K[layer, head].T / (model.cfg.d_head ** 0.5)
            # submatrix_of_full_QK_matrix: Float[Tensor, "batch seqK d_vocab"] = W_EE_toks @ W_QK.T @ W_U
            # E_sq: Int[Tensor, "batch seqK K_semantic"] = submatrix_of_full_QK_matrix.topk(K_semantic, dim=-1).indices
            W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
            submatrix_of_full_OV_matrix: Float[Tensor, "batch seqK d_vocab"] = W_EE_toks @ W_OV @ W_U
            E_sq: Int[Tensor, "batch seqK K_semantic"] = submatrix_of_full_OV_matrix.topk(K_semantic, dim=-1, largest=False).indices
            # ! Not sure if this is reasonable; we explicitly make sure pure copy-suppression is handled (even if the OV circuit is not amazing at this)
            # e.g. `" system"` is not the most negatively-copied token when we attend to `" system"`, but we're including it anyway
            E_sq_contains_self: Bool[Tensor, "batch seqK"] = (E_sq == toks[:, :, None]).any(-1)
            E_sq[..., -1] = t.where(E_sq_contains_self, E_sq[..., -1], toks)
            model_results.E_sq[layer, head] = E_sq

            # Now, for each (batch, seqQ), we want the logits for all the tokens in E_sq[batch, :, :], so we can pick the best K_unembed of those
            # i.e. logits_for_E_sq[b, sQ, sK, K_sem] = logit_lens[b, sQ, E_sq[b, sK, K_sem]]
            b = einops.repeat(t.arange(batch_size), "b -> b sQ sK K_sem", sQ=seq_len, sK=seq_len, K_sem=K_semantic)
            sQ = einops.repeat(t.arange(seq_len), "sQ -> b sQ sK K_sem", b=batch_size, sK=seq_len, K_sem=K_semantic)
            E_sq_repeated = einops.repeat(E_sq, "b sK K_sem -> b sQ sK K_sem", sQ=seq_len)
            assert b.shape == sQ.shape == E_sq_repeated.shape
            logits_for_E_sq: Float[Tensor, "batch seqQ seqK K_semantic"] = logit_lens[b, sQ, E_sq_repeated]
            assert logits_for_E_sq.shape == (batch_size, seq_len, seq_len, K_semantic)
            # We then want to make sure that the causal mask has been applied, because no pair (S, Q) for destination token D should have S < D
            sQ = einops.repeat(t.arange(seq_len), "sQ -> b sQ sK K_sem", b=batch_size, sK=seq_len, K_sem=K_semantic)
            sK = einops.repeat(t.arange(seq_len), "sK -> b sQ sK K_sem", b=batch_size, sQ=seq_len, K_sem=K_semantic)
            logits_for_E_sq = t.where(sQ >= sK, logits_for_E_sq, t.full_like(logits_for_E_sq, -1e9))
            
            # Now we find the top K_unembed logit values of pairs (S, Q) for each destination token D
            # (we get the values so that we can create a boolean mask from them)
            top_logits_for_E_sq: Float[Tensor, "batch seqQ K_unembed"] = logits_for_E_sq.flatten(-2, -1).topk(dim=-1, k=K_unembed).values
            top_logits_borderline: Float[Tensor, "batch seqQ 1 1"] = top_logits_for_E_sq[..., [[-1]]]
            top_toks_for_E_sq: Int[Tensor, "N 4"] = t.nonzero(logits_for_E_sq >= top_logits_borderline)
            # model_results.misc[(layer, "logits_for_E_sq")] = logits_for_E_sq.clone()
            # model_results.misc[(layer, "top_toks_for_E_sq")] = top_toks_for_E_sq
            # model_results.misc[(layer, "logit_lens")] = logit_lens
            # Above is a nice trick: each row is (b, sQ, sK, Ks), and E_sq[b, sK, Ks] is the token ID whose unembedding direction we want to preserve when moving from sK->sQ
            # Here is also a nice trick - get the earliest possible indices I can insert the vectors into (so that I don't overwrite any). It means we can slice the vectors so 
            # that there are fewer than `K_unembed` (which is the theoretical upper limit but in practice we don't get anywhere near that because for each D, the (S, Q) pairs 
            # are distributed over several different S's
            b, sQ, sK, Ks = top_toks_for_E_sq.unbind(-1)
            Ku = (top_toks_for_E_sq[None, :, :3] == top_toks_for_E_sq[:, None, :3]).all(-1).cumsum(-1).diag() - 1
            # Now we can get the projection directions!
            unembeddings_for_projection: Float[Tensor, "batch seqQ seqK K_unembed d_model"] = t.zeros(
                (batch_size, seq_len, seq_len, Ku.max().item()+1, model.cfg.d_model)
            ).to(current_device)
            unembeddings: Float[Tensor, "N d_model"] = W_U.T[E_sq[b, sK, Ks]]
            unembeddings_for_projection[b, sQ, sK, Ku, :] = unembeddings
            # Do G-S orthonormalization (note that we need to tranpose to get the `nums` dimension (which is K_unembed) at the end, then we transpose back)
            model_results.unembedding_projection_dirs[layer, head] = gram_schmidt(unembeddings_for_projection.transpose(-2, -1))


    if verbose:
        print(f"{'Computing logit lens (& CSPA)':<41} ... {time.time()-t0:.2f}s")
        print(f"{'Computing misc things':<41} ... ", end="\r")
        t0 = time.time()

    for (layer, head) in negative_heads:
        
        # Get output norms for value-weighted attention
        model_results.out_norm[layer, head] = einops.reduce(
            (model_results.v[layer, head] @ model.W_O[layer, head]).pow(2), 
            "batch seq d_model -> batch seq", "sum"
        ).sqrt()

        # Calculate the thing we'll be subbing in for mean ablation
        model_results.result_mean[layer, head] = einops.reduce(
            model_results.result[layer, head], 
            "batch seq d_model -> d_model", "mean"
        )


    # ! Now for the big ones: the logits, loss, and direct effects
    # For each head, there are 12 different logit terms: 3 effects (direct/indirect/both), 2 ablation modes (mean/zero), 2 final layernorm modes (frozen/unfrozen)
    # I need to get the final value of the residual stream (pre-LN) for each of the 3*2*1 = 6 modes, then I can directly compute all 12 using model_results.scale / normalization
    # I calculate "both" by doing patching at the output of the head, "direct" by patching at resid_post final, and "indirect" by subtracting the two
    # What about DLA? For each head, there are 4 different kinds (mean/zero, frozen/unfrozen), and so I also store these at the same time
    # What about loss? We have 12 different losses, one for each of the 12 logit terms.
    if verbose:
        print(f"{'Computing misc things':<41} ... {time.time()-t0:.2f}s")
        print(f"{'Computing logits, loss and direct effects':<41} ... ", end="\r")
        t0 = time.time()

    resid_post_orig = model.hook_dict[utils.get_act_name("resid_post", model.cfg.n_layers-1)].ctx.pop("orig")


    def freeze_head_z(
        z: Float[Tensor, "batch seq n_heads d_model"],
        hook: HookPoint,
        head: int,
    ):
        '''
        Freeze a head's output to what it was before (we stored this in model results).
        '''
        z[:, :, head] = model_results.z[hook.layer(), head]
        return z


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
    ):
        '''
        See docstring at the top of this function for details on what this ablation looks like. Summary: for each
        destination token D, we have up to K^2 pairs of terms (U(p), E(p,s)), and if E(p,s) exists in context then
        we project the result vector moved from that source token onto the unembedding U(p). Thanks to my implementation
        of Gram-Schmidt orthogonalization, this can still be done in a batched way even if the size of the subspaces
        we're projecting onto isn't constant over all tokens.
        
        We also store the ablated values in context (so that we can use them later to compute direct effects).

        The actual value we replace the attention head with is just the ablated values.
        '''
        assert ablation_type in ["mean", "zero"]
        layer = hook.layer()

        # Get the results pre attn (i.e. v @ W_O), and also get the ablated results.
        # Note, we ablate after multiplying by attention rather than before. This is
        # because ablating should remove all signals: attention (QK) and values (OV).
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
        result_ablated = einops.reduce(result_post_attn_ablated, "batch seqQ seqK d_model -> batch seqQ d_model", "sum")

        # Now we perform the projection we subtract the mean before projecting, i.e. we're doing Proj(x - x_mean) + x_mean rather than Proj(x). Projecting in these
        # 2 ways is equivalent to mean-ablation-based projecting vs. zero-ablation-based projecting (we want mean ablation!).
        result_post_attn_projections = result_post_attn.clone()
        result_post_attn_projections = project(
            vectors = result_post_attn_projections - result_post_attn_ablated,
            proj_directions = model_results.unembedding_projection_dirs[layer, head],
            gs = False, # We've already done GS orthonormalization on the projection directions
        ) + result_post_attn_ablated

        result_new = einops.reduce(result_post_attn_projections, "batch seqQ seqK d_model -> batch seqQ d_model", "sum")

        # Store in hook context
        hook.ctx[("result_CSPA", head)] = result_new.clone()

        # Ablate the head the normal way (for the indirect effect)
        result[:, :, head] = result_ablated
        return result



    # For each head:
    for layer, head in negative_heads:

        # For each ablation mode:
        for ablation_type in ["mean", "zero"]:

            # ! Calculate new logits, for the "both" intervention effect

            # Calculate "both" for unfrozen layernorm, by replacing this head's output with the ablated version and doing forward pass
            # Also calculate "both" for frozen layernorm, by directly computing it from the ablated value of the residual stream (which we get from patching at attn head result)
            model.add_hook(utils.get_act_name("result", layer), partial(ablate_head_result, head=head, ablation_type=ablation_type))
            resid_post_both_ablated, logits_both_ablated = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer, return_type="both")
            model_results.logits[("both", "unfrozen", ablation_type)][layer, head] = logits_both_ablated
            model_results.logits[("both", "frozen", ablation_type)][layer, head] = (resid_post_both_ablated / model_results.scale) @ model.W_U + model.b_U

            # Do exactly the same thing, just with a different (fancier!) form of ablation which preserves the pure copy-suppression mechanism.
            if cspa:
                model.add_hook(utils.get_act_name("result", layer), partial(ablate_head_result_preserve_copy_suppression_mechanism, head=head, ablation_type=ablation_type))
                model.add_hook(utils.get_act_name("resid_post", model.cfg.n_layers-1), partial(cache_resid_post, key="both"))
                resid_both_CSPA, logits_both_CSPA = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer, return_type="both")
                model_results.logits[("both", "unfrozen", ablation_type, "CS")][layer, head] = logits_both_CSPA
                model_results.logits[("both", "frozen", ablation_type, "CS")][layer, head] = (resid_both_CSPA / model_results.scale) @ model.W_U + model.b_U

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
            if cspa:
                resid_post_DLA_CS = model_results.result[layer, head] - model.hook_dict[utils.get_act_name("result", layer)].ctx.pop(("result_CSPA", head))
                resid_post_direct_CSPA = resid_post_orig - resid_post_DLA_CS
                logits_direct_ablated_frozen_CS = (resid_post_direct_CSPA / model_results.scale) @ model.W_U + model.b_U
                logits_direct_ablated_unfrozen_CS = (resid_post_direct_CSPA / resid_post_direct_CSPA.std(-1, keepdim=True)) @ model.W_U + model.b_U
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
            if cspa:
                resid_post_indirect_CSPA = resid_post_orig + (resid_both_CSPA - resid_post_direct_CSPA)
                model_results.logits[("indirect", "frozen", ablation_type, "CS")][layer, head] = (resid_post_indirect_CSPA / model_results.scale) @ model.W_U + model.b_U
                model_results.logits[("indirect", "unfrozen", ablation_type, "CS")][layer, head] = (resid_post_indirect_CSPA / resid_post_indirect_CSPA.std(-1, keepdim=True)) @ model.W_U + model.b_U

            # ! Calculate the effect of ablating the indirect effect minus 11.10, i.e. ablating all indirect paths which DON'T involve head 11.10

            if (layer, head) == (10, 7):
                # For this case, we perform 3 interventions: (1) ablate output of head 10.7 (i.e. actually change attn result), (2) freeze 11.10's input, (3) recover 10.7's full output at resid post
                later_layer, later_head = (11, 10)
                effect = f"indirect (excluding {later_layer}.{later_head})"
                model.add_hook(utils.get_act_name("result", layer), partial(ablate_head_result, head=head, ablation_type=ablation_type)) # (1)
                model.add_hook(utils.get_act_name("z", later_layer), partial(freeze_head_z, head=later_head)) # (2)
                # Unlike the last 2 times we performed `model_fwd_pass_from_resid_pre`, we're calculating logits explicitly rather than returning them (cause we need to add DLA back in)
                resid_post_pre_step3 = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer, return_type="resid_post") # (3) (and below)
                resid_post_post_step3 = resid_post_pre_step3 + resid_post_DLA
                model_results.logits[(effect, "frozen", ablation_type)][layer, head] = (resid_post_post_step3 / model_results.scale) @ model.W_U + model.b_U
                model_results.logits[(effect, "unfrozen", ablation_type)][layer, head] = (resid_post_post_step3 / resid_post_post_step3.std(-1, keepdim=True)) @ model.W_U + model.b_U

            # ! Calculate new loss (and while doing this, get all the tensors which tell us if this example is copy suppression)

            # For each of these six logits, calculate loss differences (for regular ablation, and CS-preserving ablation)
            for effect in ["both", "direct", "indirect"] + ([f"indirect (excluding {later_layer}.{later_head})"] if (layer, head) == (10, 7) else []):
                for ln_mode in ["frozen", "unfrozen"]:

                    # Get the logits, compute and store the corresponding loss
                    logits = model_results.logits[(effect, ln_mode, ablation_type)][layer, head]
                    loss_ablated = model.loss_fn(logits, toks, per_token=True)
                    model_results.loss_diffs[(effect, ln_mode, ablation_type)][layer, head] = loss_ablated - model_results.loss_orig

                    if cspa:
                        # Get the logits for CS-preserving ablation, compute and store the corresponding loss
                        logits_CS = model_results.logits[(effect, ln_mode, ablation_type, "CS")][layer, head]
                        loss_CS = model.loss_fn(logits_CS, toks, per_token=True)
                        model_results.loss_diffs[(effect, ln_mode, ablation_type, "CS")][layer, head] = loss_CS - model_results.loss_orig

                        # Store the results
                        model_results.is_copy_suppression[(effect, ln_mode, ablation_type)][layer, head] = {
                            "L_ORIG": model_results.loss_orig,
                            "L_ABL": loss_ablated,
                            "L_CS": loss_CS,
                        }

    if verbose:
        print(f"{'Computing logits, loss and direct effects':<41} ... {time.time()-t0:.2f}s")

    model_results.clear()

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

    return model_results