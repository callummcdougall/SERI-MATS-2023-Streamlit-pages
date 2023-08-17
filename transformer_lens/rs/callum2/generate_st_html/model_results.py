# This file contains functions to get results from my model. I use them to generate the visualisations found on the Streamlit pages.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
import itertools
from typing import Dict, Any, Tuple, List, Optional, Literal
from transformer_lens import HookedTransformer, utils
from functools import partial
import einops
from dataclasses import dataclass
import gc
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int, Float, Bool
import torch as t
from collections import defaultdict
import time
from tqdm import tqdm
from torch import Tensor

from transformer_lens.rs.callum2.cspa.cspa_utils import (
    devices_are_equal,
    kl_div,
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
    ' at', ' of', 'to', ' now', "'s", 'The', ".", ",", "?", "!", " '",
    # Need to go through these words and add more to them, I suspect this list is minimal
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
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class LayerResults:
    data: Dict[int, Tensor]
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
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


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
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


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
        self.kl_divs: DictOfHeadResults = DictOfHeadResults()

        # We need the result vectors for each head, we'll use it for patching
        self.z: HeadResults = HeadResults()
        self.result: HeadResults = HeadResults()
        self.result_mean: HeadResults = HeadResults()

        # We need data for attn & weighted attn (and for projecting result at source token unembeds)
        self.pattern: HeadResults = HeadResults()
        self.v: HeadResults = HeadResults()
        self.out_norm: HeadResults = HeadResults()

        # resid_pre for each head (to get unembed components)
        self.resid_pre: LayerResults = LayerResults()
        
        # Layernorm scaling factors, pre-attention layer for each head & at the very end
        self.scale_attn: LayerResults = LayerResults()
        self.scale: Tensor = t.empty(0)

        # This is the "prediction-attention" part, and it's how we perform copy-suppression preserving ablation
        self.logit_lens: LayerResults = LayerResults()

        # This is to store random crap!
        self.misc: dict = {}


    def clear(self):
        """Empties all imtermediate results we don't need."""
        if "result" in self.keys(): del self.result
        if "result_mean" in self.keys(): del self.result_mean
        if "resid_pre" in self.keys(): del self.resid_pre
        if "v" in self.keys(): del self.v
        if "scale_attn" in self.keys(): del self.scale_attn
        t.cuda.empty_cache()

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def to(self, device):
        for key, value in self.items():
            assert any([isinstance(value, cls) for cls in [Tensor, HeadResults, LayerResults, DictOfHeadResults, dict]]), f"Unknown type for {key}: {type(value)}"
            if not isinstance(value, dict):
                self[key] = value.to(device)
        return self

    def concat_results(self, other: "ModelResults", keep_logits: bool) -> "ModelResults":
        '''
        Given 2 model results objects, returns their concatenation along the batch dimension. Only keeps the non-cleared things.
        '''
        self.clear()
        # Create new results, and make sure it's on the correct device (self.loss_orig will always be defined by this point)
        new = ModelResults().to(self.loss_orig.device)
        if not(keep_logits): # todo - find a better way to do this, it's janky as all hell right now 
            del new.logits
            del new.logits_orig
            del new.dla
        for k, v in self.items():
            # All tensors have first dimension equal to batch dim; we concat along this
            if isinstance(v, Tensor):
                new[k] = t.cat([self[k], other[k]])
            # All HeadResults or LayerResults are basically dicts where the values are tensors of shape (batch, ...)
            elif isinstance(v, HeadResults):
                new[k] = HeadResults({head: t.cat([v[head], other[k][head]]) for head in v.keys()})
            elif isinstance(v, LayerResults):
                new[k] = LayerResults({layer: t.cat([v[layer], other[k][layer]]) for layer in v.keys()})
            # All DictOfHeadResults are dicts where the values are HeadResults
            elif isinstance(v, DictOfHeadResults):
                new[k] = DictOfHeadResults()
                for key, headresults in v.items():
                    new[k][key] = HeadResults({head: t.cat([results, other[k][key][head]]) for head, results in headresults.items()})
            # Lastly, the misc dict!
            elif isinstance(v, dict):
                new[k] = {**v, **other[k]}
            else:
                raise ValueError(f"Unknown type for {k}: {type(v)}")
        new.clear()
        assert sorted(new.keys()) == sorted(self.keys()), f"Some keys were not concatenated. New - old = {set(new.keys())-set(self.keys())}, old - new = {set(self.keys())-set(new.keys())}."
        return new



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
    assert proj_directions.shape[:-1] == vectors.shape
    assert proj_directions.shape[-1] <= 30, "Shouldn't do too many vectors, GS orth might be computationally heavy I think?"

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
    scale: Optional[Float[Tensor, "batch seq"]] = None,
) -> Float[Tensor, "batch seq d_vocab"]:
    '''
    Performs a forward pass starting from an intermediate point.

    For instance, if layer=10, this will apply the TransformerBlocks from
    layers 10, 11 respectively, then ln_final and unembed.

    Also, if return_resid_post = True, then it just returns the final value
    of the residual stream, i.e. omitting ln_final and unembed.
    '''
    assert return_type in ["logits", "resid_post", "both"]

    resid = resid_pre.clone()
    for i in range(layer, model.cfg.n_layers):
        resid = model.blocks[i](resid)
    
    if (return_type == "resid_post"): return resid

    # Frozen or unfrozen layernorm?
    if scale is None:
        resid_scaled = model.ln_final(resid)
        logits = model.unembed(resid_scaled)
    else:
        resid_scaled = (resid / resid.std(dim=-1, keepdim=True)) @ model.W_U + model.b_U
    
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
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    use_cuda: bool = False,
    verbose: bool = False,
    effective_embedding: str = "W_E (including MLPs)",
    keep_logits: bool = True,
    keep_seq_dim_when_mean_ablating: bool = True,
) -> ModelResults:
    '''
    result_mean is the vector we'll use for ablation, if supplied. It'll map e.g. (10, 7) to 
    the mean result vector for each seqpos (hopefully seqpos is larger than that for toks).

    Logits take up the most space by far (because d_vocab dimension), so we can set keep_logits=False
    to delete those from the model. Usually we don't actually want the logits; we want something that
    is derived from them.
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
        print(f"{'Running forward pass':<24} ... ", end="\r")
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

    def cache_resid_post(resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint, key: str):
        hook.ctx[key] = resid_post

    def cache_scale_attn(scale: Float[Tensor, "batch seq *head 1"], hook: HookPoint):
        scale_corrected = scale.clone()
        if scale_corrected.ndim == 4:
            # TODO - figure out exactly when we have 4 dims not 1 (I assume 1 per head, for causal interventions)
            scale_corrected = scale_corrected[:, :, 0]
        model_results.scale_attn[hook.layer()] = scale_corrected


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
    # We'll eventually need to freeze head 11.10's output, even if it's not one of the ones we're measuring
    if (11, 10) not in negative_heads:
        model.add_hook(utils.get_act_name("z", 11), partial(cache_head_z, head=10))
    # We also need some things at the very end of the model
    model.add_hook(utils.get_act_name("resid_post", model.cfg.n_layers-1), partial(cache_resid_post, key="orig"))
    model.add_hook(utils.get_act_name("scale"), cache_scale)

    # Run the forward pass, to cache all values (and get original logits)
    model_results.logits_orig, model_results.loss_orig = model.forward(toks, return_type="both", loss_per_token=True)
    model.reset_hooks(including_permanent=True, clear_contexts=False)


    # ! Get all the tokens in context, and what unembeddings most attend to them, i.e. the pairs (S, Q)
    if verbose:
        print(f"{'Running forward pass':<24} ... {time.time()-t0:.2f}s")
        print(f"{'Computing model results':<24} ... ", end="\r")
        t0 = time.time()

    for layer, head in negative_heads:

        # We get the logit lens, so that (for each head) we can figure out what to project onto
        resid_pre = model_results.resid_pre[layer]
        scale = model_results.scale_attn[layer]
        resid_pre_scaled: Float[Tensor, "batch seqQ d_model"] = resid_pre / scale
        assert t.all((resid_pre_scaled.norm(dim=-1) - model.cfg.d_model ** 0.5) < 1e-4)
        logit_lens: Float[Tensor, "batch seqQ d_vocab"] = resid_pre_scaled @ W_U
        model_results.logit_lens[layer] = logit_lens

    for (layer, head) in negative_heads:
        
        # Get output norms for value-weighted attention
        model_results.out_norm[layer, head] = einops.reduce(
            (model_results.v[layer, head] @ model.W_O[layer, head]).pow(2), 
            "batch seq d_model -> batch seq", "sum"
        ).sqrt()

        # Calculate the thing we'll be subbing in for mean ablation
        # * Note - we preserve the destination position when we take our mean, to preserve positional info, and for consistency with CSPA
        if result_mean is None:
            head_result_mean = einops.reduce(
                model_results.result[layer, head], 
                "batch seqQ d_model -> seqQ d_model", "mean"
            )
            # If instructed, we also want to average over the sequence dimension
            if not(keep_seq_dim_when_mean_ablating):
                head_result_mean[:] = einops.reduce(
                    head_result_mean,
                    "seqQ d_model -> d_model", "mean"
                )
        else:
            head_result_mean = result_mean[(layer, head)]
            if head_result_mean.ndim == 2:
                # In this case, the result mean is shape (seq, d_model)
                assert keep_seq_dim_when_mean_ablating, "Your `result_mean` vector has a sequence dimension, but you set `keep_seq_dim_when_mean_ablating=False`."
                assert head_result_mean.shape[0] >= seq_len
                head_result_mean = head_result_mean[:seq_len]
            else:
                # In this case, the result mean is shape (d_model,)
                assert not(keep_seq_dim_when_mean_ablating), "Your `result_mean` vector has no sequence dimension, but you set `keep_seq_dim_when_mean_ablating=True`."
                head_result_mean = einops.repeat(head_result_mean, "d_model -> seqQ d_model", seqQ=seq_len)

        model_results.result_mean[layer, head] = head_result_mean

    # ! Now for the big ones: the logits, loss, and direct effects
    # TODO - rewrite this big comment, it's a bit outdated
    # For each head, there are 12 different logit terms: 3 effects (direct/indirect/both), 2 ablation modes (mean/zero), 2 final layernorm modes (frozen/unfrozen)
    # I need to get the final value of the residual stream (pre-LN) for each of the 3*2*1 = 6 modes, then I can directly compute all 12 using model_results.scale / normalization
    # I calculate "both" by doing patching at the output of the head, "direct" by patching at resid_post final, and "indirect" by subtracting the two
    # What about DLA? For each head, there are 4 different kinds (mean/zero, frozen/unfrozen), and so I also store these at the same time
    # What about loss? We have 12 different losses, one for each of the 12 logit terms.

    resid_post_orig: Tensor = model.hook_dict[utils.get_act_name("resid_post", model.cfg.n_layers-1)].ctx.pop("orig")

    def freeze_head_z(z: Float[Tensor, "batch seq n_heads d_model"], hook: HookPoint, head: int):
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

        Note, we'll be using the thing cached here as the vector to ablate with in the next block. It's either going to be 
        the same as model_results.result_mean, or it'll be zero.
        '''
        assert ablation_type in ["mean", "zero"]

        if ablation_type == "zero":
            result[:, :, head] = t.zeros_like(result[:, :, head])
        elif ablation_type == "mean":
            result[:, :, head] = model_results.result_mean[hook.layer(), head]

        # Store in hook context, and return
        hook.ctx[("result_ablated", head)] = result[:, :, head]
        return result


    def get_logits(
        resid: Float[Tensor, "batch seq d_model"],
        ln_type: Literal["frozen", "unfrozen"],
        bias: bool = True
    ):
        '''Helper function, because I find myself writing this equation out a lot.'''
        assert ln_type in ["frozen", "unfrozen"]
        scale = model_results.scale if (ln_type == "frozen") else resid.std(-1, keepdim=True)
        bias = model.b_U if bias else 0.0
        return (resid / scale) @ model.W_U + bias


    # For each head:
    for layer, head in negative_heads:

        # Get a list of all the ablation effects we're iterating through, so we can eventually calculate all the losses & KL divergences
        effects = ["both", "direct", "indirect"]
        # Always mean ablation (no longer doing "zero" too)
        ablation_type = "mean"
        # Get result hook name as shorthand
        result_hook_name = utils.get_act_name("result", layer)

        # ! Calculate new logits, for the "both" intervention effect

        # We intervene at the head output, and use the `model_fwd_pass_from_resid_pre` function. Note that this returns logits and resid_post here;
        # the former already has unfrozen layernorm applied, and the latter we manually apply frozen layernorm to get frozen logits
        model.add_hook(result_hook_name, partial(ablate_head_result, head=head, ablation_type=ablation_type))
        resid_post_both_ablated = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer, return_type="resid_post")
        model.reset_hooks(clear_contexts=False)
        model_results.logits[("both", "frozen", ablation_type)][layer, head] = get_logits(resid_post_both_ablated, "frozen")
        model_results.logits[("both", "unfrozen", ablation_type)][layer, head] = get_logits(resid_post_both_ablated, "unfrozen")

        # ! Calculate new logits for the "direct" intervention effect, and calculate the DLA

        # Rather than running forward passes, we manually compute the direct effect, and new layernorm scale factors
        # `head_DLA_in_resid` is (residual stream direct contribution of this head) - (mean ablated direct contribution)
        head_result_mean_ablated = model.hook_dict[result_hook_name].ctx.pop(("result_ablated", head))
        head_DLA_in_resid: Tensor = model_results.result[layer, head] - head_result_mean_ablated
        resid_post_direct_ablated = resid_post_orig - head_DLA_in_resid
        logits_direct_ablated_frozen = get_logits(resid_post_direct_ablated, "frozen")
        logits_direct_ablated_unfrozen = get_logits(resid_post_direct_ablated, "unfrozen")
        model_results.logits[("direct", "frozen", ablation_type)][layer, head] = logits_direct_ablated_frozen
        model_results.logits[("direct", "unfrozen", ablation_type)][layer, head] = logits_direct_ablated_unfrozen
        # This is also where we compute the direct logit attribution for this head
        model_results.dla[("frozen", ablation_type)][layer, head] = get_logits(head_DLA_in_resid, "frozen", bias=False)
        model_results.dla[("unfrozen", ablation_type)][layer, head] = model_results.logits_orig - logits_direct_ablated_unfrozen

        # ! Calculate new logits for the "indirect" intervention effect

        # We do this by taking the residual stream at the end (it's what we get when both are ablated, but with DLA added back in)
        resid_post_indirect_ablated = resid_post_both_ablated + head_DLA_in_resid
        model_results.logits[("indirect", "frozen", ablation_type)][layer, head] = get_logits(resid_post_indirect_ablated, "frozen")
        model_results.logits[("indirect", "unfrozen", ablation_type)][layer, head] = get_logits(resid_post_indirect_ablated, "unfrozen")

        # ! Calculate the effect of ablating the indirect effect minus 11.10, i.e. ablating all indirect paths which DON'T involve head 11.10

        if layer < 11:
            # Here, we ablate the head's indirect path, but freeze 11.10's output so we don't count this path. Then we add 10.7 back at the end, for the direct effect
            later_layer, later_head = (11, 10)
            effect = f"indirect (excluding {later_layer}.{later_head})"
            effects.append(effect)
            model.add_hook(result_hook_name, partial(ablate_head_result, head=head, ablation_type=ablation_type))
            model.add_hook(utils.get_act_name("z", later_layer), partial(freeze_head_z, head=later_head))
            # This gives us the result of ablating everything except indirect path -> 11.10, so we add back in DLA for 10.7 to ablate only (indirect excl. 11.10)
            resid_post_both_excl_L11H10_ablated = model_fwd_pass_from_resid_pre(model, model_results.resid_pre[layer], layer, return_type="resid_post")
            model.reset_hooks(clear_contexts=False)
            resid_post_indirect_excl_L11H10_ablated = resid_post_both_excl_L11H10_ablated + head_DLA_in_resid
            model_results.logits[(effect, "frozen", ablation_type)][layer, head] = get_logits(resid_post_indirect_excl_L11H10_ablated, "frozen")
            model_results.logits[(effect, "unfrozen", ablation_type)][layer, head] = get_logits(resid_post_indirect_excl_L11H10_ablated, "unfrozen")

        # ! Calculate new loss (and while doing this, get all the tensors which tell us if this example is copy suppression)

        # For each of these six logits, calculate loss differences
        for effect, ln_mode in itertools.product(effects, ["frozen", "unfrozen"]):
            # Get the logits, compute and store the corresponding loss
            logits_ablated = model_results.logits[(effect, ln_mode, ablation_type)][layer, head]
            loss_ablated = model.loss_fn(logits_ablated, toks, per_token=True)
            model_results.loss_diffs[(effect, ln_mode, ablation_type)][layer, head] = loss_ablated - model_results.loss_orig
            model_results.kl_divs[(effect, ln_mode, ablation_type)][layer, head] = kl_div(model_results.logits_orig, logits_ablated)

    if verbose:
        print(f"{'Computing model results':<24} ... {time.time()-t0:.2f}s")

    model_results.clear()

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

    if not(keep_logits):
        del model_results.logits_orig
        del model_results.logits
        del model_results.dla

    model.reset_hooks(including_permanent=True)
    return model_results



# TODO - remove all default arguments on the non-batched version of this function (don't let anything get missed)

def get_model_results_batched(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    max_batch_size: int,
    negative_heads: List[Tuple[int, int]],
    use_cuda: bool = False,
    store_in_cuda: bool = False,
    verbose: bool = False,
    effective_embedding: str = "W_E (including MLPs)",
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    keep_logits: bool = True,
    keep_seq_dim_when_mean_ablating: bool = True,
) -> ModelResults:
    '''
    Does same thing as `get_model_results`, but does it in a batched way (avoiding cuda errors).
    '''
    chunks = toks.shape[0] // max_batch_size

    device = t.device("cuda" if use_cuda else "cpu")
    result_mean = {k: v.to(device) for k, v in result_mean.items()}

    orig_model_device = str(next(iter(model.parameters())).device)
    orig_toks_device = str(toks.device)
    target_device = "cuda" if use_cuda else "cpu"
    if not devices_are_equal(orig_model_device, target_device):
        model = model.to(target_device)
    if not devices_are_equal(orig_toks_device, target_device):
        toks = toks.to(target_device)

    model_results = None

    for i, _toks in enumerate(t.chunk(toks, chunks=chunks)):
        if verbose:
            if i == 0: bar = tqdm(total=chunks, desc=f"Batch {i+1}/{chunks}, shape {tuple(_toks.shape)}")
            t0 = time.time()

        model_results_new = get_model_results(
            model,
            _toks,
            negative_heads,
            use_cuda=use_cuda,
            effective_embedding=effective_embedding,
            result_mean=result_mean,
            keep_logits=keep_logits,
            keep_seq_dim_when_mean_ablating=keep_seq_dim_when_mean_ablating,
        )
        if not(store_in_cuda): model_results_new = model_results_new.to("cpu")
        model_results = model_results_new if (model_results is None) else model_results.concat_results(model_results_new, keep_logits=keep_logits)

        gc.collect()
        t.cuda.empty_cache()

        if verbose:
            t0 = time.time() - t0
            bar.update()
            bar.set_description(f"Batch {i+1}/{chunks}, shape {tuple(_toks.shape)}, time {t0:.2f}s")
    
    return model_results
