# This file contains functions to get results from my model. I use them to generate the visualisations found on the Streamlit pages.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
for root_dir in [
    os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts", # For Arthur's branch
    "/app/seri-mats-2023-streamlit-pages/explore_prompts", # For Streamlit page (public)
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/explore_prompts", # For Arthur's branch
    os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0] + "SERI-MATS-2023-Streamlit-pages/explore_prompts", # For Arthur's branch
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
    data: Dict[Head, Tensor]
    '''
    Stores results for each head. This is just a standard dictionary, but you can index it like [layer, head]
    (this is just a convenience thing, so I don't have to wrap the head in a tuple).
    '''
    def __init__(self, data=None):
        self.data = data or {}
    def __getitem__(self, layer_and_head: Head) -> Tensor:
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

        # This is the "prediction-attention" part, and it's how we perform copy-suppression preserving ablation
        self.logit_lens: LayerResults = LayerResults()
        self.U_p: HeadResults = HeadResults()
        self.E_ps: HeadResults = HeadResults()

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
    Kp: int = 10,
    Ks: int = 10,
) -> ModelResults:
    '''
    Short explpanation of the copy-suppression preserving ablation (hereby CSPA):

        (1) For each destination token D, look at the logit lens at D. Pick the top Kp tokens being
            predicted at that point.

        (2) For each of these tokens U(1), ..., U(Kp), look at the full QK matrix (i.e. the matrix
            W_U @ W_QK @ W_EE.T) to figure out what tokens the unembedding of U(1) will pay attention
            to. Pick the top Ks of these, for each token. Call these E(p,s) for each U(p). Note that
            the set E(p,s) for a given s will usually include U(p), as well as things very semantically
            similar to U(p) - more on this later.

        (3) We now have a maximum of Kp*Ks tokens. For each source token S, we take its output vector
            that gets moved to D (i.e. the vector Attn[D,S] * v[S] @ W_OV) and do the following:

                If S isn't in the set of E(i, j), we mean-ablate this vector.

                If S is in the set of E(i, j), we project this vector onto the span of all the vectors
                U(p) s.t. S = E(p,s) for some j.

    Why does this capture the copy-suppression mechanism?

    Our theory of copy suppression is something like "a token P is being predicted to come after D.
    P is semantically similar to S, which appeared earlier in context. We attend from D back to S, 
    and the result is that we suppress the prediction P.

    Here are a few examples:

        "All's fair in love and war."
            D = " and"
            P = " love"
            S = " love"
            The model is incorrectly predicting that " and" follows the first " love". The unembedding
            for " love" causes it to attend from " and" back to " love", and negatively copy it.

        "Berkeley ... University of California..."
            D = " of"
            P = " Berkeley"
            S = "keley"
            The model is incorrectly predicting "University of Berkeley". The unembedding of " Berkeley"
            causes it to attend from " of" back to "keley", and it negatively copies " Berkeley".
        
        " Pier ... at the pier"
            D = " the"
            P = " pier"
            S = " Pier"
            The model is correctly predicting "at the pier", and incorrectly predicting "at the Pier"
            (this reflects the fact that it thinks some form of "pier" comes next, but it's agnostic
            about capitalization). Both the " pier" unembedding and the " Pier" unembedding cause it
            to attend back to " Pier". Attending to " Pier" causes it to negatively copy both " pier"
            and " Pier".

    "Pure copy-suppression" is when S = T, i.e. the model is literally suppressing the token that follows
    itself. But often this isn't the case (e.g. the second example). Other situations of non-pure copy
    suppression are capitals (" pier" / " Pier") or plurals (" device" / " devices") or leading spaces.

    In our method, the tokens U(p)correspond to the predicted token P, and the tokens E(p,s) correspond to the
    tokens S which are attended to. So, we're basically capturing this notion of "semantic similarity", rather
    than resticting ourselves to looking at "pure copy suppression".

    By expanding to more than just "pure copy suppression", we run a risk. We're now projecting onto at
    most Kp*Ks = 100 different subspaces rather than just Kp subspaces. So we capture more stuff, but maybe
    lose the ability to understand the things we're capturing. To show this isn't happening, let's present
    some stats about the tokens U(p) and E(p,s) we end up with:

        > A large fraction of the top-5% (I'm guessing a bit below 50%), we have exactly one U(p) and one E(p,s)
          in context, and they're the same.
        > A large fraction of the U(p)s (I'm guessing above 50%), there's exactly one E(p,s).
        > A very large fraction of the U(p)s (I'm guessing close to 100%), the E(p,s)s which actually appear in
          context are semantically similar to U(p) via one of the following:
            - They're the same token
            - They differ by plurality/spaces/capitalization
            - They're linked via tokenization weirdness (e.g. "keley" and " Berkeley")
            - They're synonyms (e.g. " researcher" and " academic")
    '''
    from transformer_lens.rs.callum.keys_fixed import get_effective_embedding_2
    W_EE_dict = get_effective_embedding_2(model)
    W_EE = W_EE_dict["W_E (including MLPs)"]
    W_EE = W_EE / W_EE.norm(dim=-1, keepdim=True)
    W_EE0 = W_EE_dict["W_E (only MLPs)"]
    W_EE0 = W_EE0 / W_EE0.norm(dim=-1, keepdim=True)
    W_U = model.W_U
    
    FUNCTION_TOKS = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze()
    
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()

    current_device = str(next(iter(model.parameters())).device)
    if use_cuda and current_device == "cpu":
        model = model.cuda()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cpu()

    model_results = ModelResults()

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


    # ! Use the logit lens to find each of the U(p)s, and use the QK circuit to find each of the E(p,s)s
    # Also get other useful things like model_results.out_norm (for info-weighted attention) and result_mean (for mean ablation)
    if verbose: print("Computing unembedding components...")

    for layer, head in negative_heads:

        # Find out what's being predicted (i.e. the logit lens) at destination token D. Call this U_p.
        resid_pre = model_results.resid_pre[layer]
        scale = model_results.scale_attn[layer]
        resid_pre_scaled: Float[Tensor, "batch seqQ d_model"] = resid_pre / scale
        assert t.all((resid_pre_scaled.norm(dim=-1) - model.cfg.d_model ** 0.5) < 1e-4)
        logit_lens = resid_pre_scaled @ W_U
        logits_topk = logit_lens.topk(Kp, dim=-1)
        U_p: Int[Tensor, "batch seqQ Kp"] = logits_topk.indices

        # Next, get the full QK matrix, and figure out what things U_p pay most attention to. Call this E_ps.
        W_QK = model.W_Q[layer, head] @ model.W_K[layer, head].T / (model.cfg.d_head ** 0.5)
        submatrix_of_full_QK_matrix: Float[Tensor, "batch seqQ Kp d_vocab"] = W_U.T[U_p] @ W_QK @ W_EE.T
        E_ps: Int[Tensor, "batch seqK Kp Ks"] = submatrix_of_full_QK_matrix.topk(Ks, dim=-1).indices

        # Store results
        model_results.U_p[layer, head] = U_p
        model_results.E_ps[layer, head] = E_ps


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
    if verbose: print("Computing logits, loss and direct effects...")

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
        batch_size, seq_len = result.shape[:2]
        layer = hook.layer()

        U_p: Int[Tensor, "batch seqQ Kp"] = model_results.U_p[layer, head]
        E_ps: Int[Tensor, "batch seqQ Kp Ks"] = model_results.E_ps[layer, head]

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
        result_ablated = einops.reduce(result_post_attn_ablated, "batch seqQ seqK d_model -> batch seqQ d_model", "sum")

        # Goal: get an Int-type array of shape (batch, seqQ, seqK, K), where the [b, sq, sk, :]-th slice gives you 
        # all the indices of the tokens which result_post_attn[b, sq, sk] should be projected onto.
        E_ps_repeated = einops.repeat(E_ps, "batch seqQ Kp Ks -> batch seqQ seqK Kp Ks", seqK=seq_len)
        U_p_repeated = einops.repeat(U_p, "batch seqQ Kp -> batch seqQ seqK Kp", seqK=seq_len)
        toks_repeated = einops.repeat(toks, "batch seqK -> batch seqQ seqK Kp Ks", seqQ=seq_len, Kp=Kp, Ks=Ks)
        toks_match_E_ps: Bool[Tensor, "batch seqQ seqK Kp Ks"] = toks_repeated == E_ps_repeated
        # For each (batch, seqQ, seqK, Kp), if one of the Ks matches then we want to project onto that U(Kp). If not, we want to ablate. So
        # we set the corresponding values of U_p_repeated to minus one where there's no match. We also want to make sure there's no match if
        # either (1) q < k (causal attention), or (2) q = first token and BOS token, or (3) k = function word.
        assert t.all(toks_match_E_ps.int().sum(-1) <= 1)
        toks_match_E_ps_for_some_s: Bool[Tensor, "batch seqQ seqK Kp"] = toks_match_E_ps.any(-1)
        causal_mask = einops.repeat(t.tril(t.ones(batch_size, seq_len, seq_len)), "batch seqQ seqK -> batch seqQ seqK Kp", Kp=Kp).bool()
        query_is_not_first_and_bos = einops.repeat(
            t.concat([toks[:, [0]] != model.tokenizer.bos_token_id, t.ones(batch_size, seq_len-1, dtype=t.bool)], dim=-1), 
            "batch seqQ -> batch seqQ seqK Kp", 
            seqK=seq_len, Kp=Kp
        )
        key_is_not_bos = einops.repeat(
            toks != model.tokenizer.bos_token_id, 
            "batch seqK -> batch seqQ seqK Kp", 
            seqQ=seq_len, Kp=Kp
        )
        key_is_not_function_word = einops.repeat(
            (toks[:, :, None] != FUNCTION_TOKS[None, None, :]).all(-1),
            "batch seqK -> batch seqQ seqK Kp", 
            seqQ=seq_len, Kp=Kp
        )
        U_p_repeated: Int[Tensor, "batch seqQ seqK Kp"] = t.where(
            toks_match_E_ps_for_some_s & causal_mask & query_is_not_first_and_bos & key_is_not_bos & key_is_not_function_word,
            U_p_repeated,
            -1
        )
        # Now we have a collection of token indices and -1s, we get the corresponding unembedding vectors and then zero them appropriately
        unembeddings_to_project_along = W_U.T[U_p_repeated]
        unembeddings_to_project_along: Float[Tensor, "batch seqQ seqK Kp d_model"] = t.where(
            einops.repeat(U_p_repeated != -1, "batch seqQ seqK Kp -> batch seqQ seqK Kp d_model", d_model=model.cfg.d_model),
            unembeddings_to_project_along,
            t.zeros_like(unembeddings_to_project_along),
        )
        if head == 7:
            model_results.misc["U_p_repeated"] = U_p_repeated
        # Finally, we have the directions which we wanted! We'll now project along them. We also need to rearrange, to match `project` function
        proj_directions = einops.rearrange(unembeddings_to_project_along, "batch seqQ seqK Kp d_model -> batch seqQ seqK d_model Kp")

        # Note we subtract the mean before projecting, i.e. we're doing Proj(x - x_mean) + x_mean rather than Proj(x). Projecting in these
        # 2 ways is equivalent to mean-ablation-based projecting vs. zero-ablation-based projecting (we want mean ablation!).
        result_post_attn_projections = result_post_attn.clone()
        result_post_attn_projections = project(
            vectors = result_post_attn_projections - result_post_attn_ablated,
            proj_directions = proj_directions,
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
            model.add_hook(utils.get_act_name("resid_post", model.cfg.n_layers-1), partial(cache_resid_post, key="both"))
            model_results.logits[("both", "unfrozen", ablation_type)][layer, head] = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer)
            resid_post_both_ablated = model.hook_dict[utils.get_act_name("resid_post", model.cfg.n_layers-1)].ctx.pop("both")
            model_results.logits[("both", "frozen", ablation_type)][layer, head] = (resid_post_both_ablated / model_results.scale) @ model.W_U + model.b_U

            # Do exactly the same thing, just with a different (fancier!) form of ablation which preserves the pure copy-suppression mechanism.
            model.add_hook(utils.get_act_name("result", layer), partial(ablate_head_result_preserve_copy_suppression_mechanism, head=head, ablation_type=ablation_type))
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
            resid_post_DLA_CS = model_results.result[layer, head] - model.hook_dict[utils.get_act_name("result", layer)].ctx.pop(("result_CSPA", head))
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
                    loss_ablated = model.loss_fn(logits, toks, per_token=True)
                    model_results.loss_diffs[(effect, ln_mode, ablation_type)][layer, head] = loss_ablated - model_results.loss_orig

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

    if verbose: print("Finishing...")

    model_results.clear()

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

    return model_results