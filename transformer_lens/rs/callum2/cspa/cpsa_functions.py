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
    ' at', ' of', 'to', ' now', "'s", 'The', ".", ",", "?", "!", # Need to go through these words and add more to them, I suspect this list is minimal
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


def get_cspa_results(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_head: Tuple[int, int],
    components_to_project: List[str] = ["v"],
    K_semantic: int = 10,
    effective_embedding: str = "W_E (including MLPs)",
    use_cuda: bool = False,
    verbose: bool = False,
) -> ModelResults:
    '''
    Short explpanation of the copy-suppression preserving ablation (hereby CSPA):

        (TODO)
    '''

    # ====================================================================
    # STEP 0: Setup
    # ====================================================================

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

    # if verbose:
    #     print(f"{'Running forward pass':<41} ... ", end="\r")
    #     t0 = time.time()
    #     print(f"{'Computing logits, loss and direct effects':<41} ... {time.time()-t0:.2f}s")



    # ====================================================================
    # STEP 1: Get effective embeddings, and semantically similar unembeddings
    # ====================================================================

    effective_embeddings = W_EE[toks] # [batch seqK d_model]


    # ====================================================================
    # STEP 2: Define hook functions
    # ====================================================================

    def hook_v(
        v_input: Float[Tensor, "batch seqK d_model"],
        hook: HookPoint,
    ):
        '''
        Projects the value input onto the effective embedding for the source token.
        '''
        v_input_mean = v_input.mean(dim=0, keepdim=True)

        return project(
            vectors = v_input - v_input_mean,
            proj_directions = effective_embeddings,
        ) + v_input_mean



    # ====================================================================
    # STEP -1: Return the results
    # ====================================================================

    # First, get clean results (also use this to get residual stream before layer 10)
    loss, cache = model.run_with_cache(
        toks,
        return_type = "logits",
        loss_per_token = True,
        names_filter = lambda name: name == utils.get_act_name("resid_pre", 10)
    )
    resid_pre = cache["resid_pre", 10]
    del cache
    # Second, get results after hooking at v
    logits_hooked = model_fwd_pass_from_resid_pre(model, resid_pre, layer=10, return_type="logits")
    loss_hooked = model.loss_fn(logits_hooked, toks, per_token=True)

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

    return {
        "loss": loss,
        "loss_v_projected": loss_hooked
    }