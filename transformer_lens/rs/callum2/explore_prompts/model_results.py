# This file contains functions to get results from my model. I use them to generate the visualisations found on the Streamlit pages.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
root_dir = os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts"
os.chdir(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

from typing import Dict, Any, Tuple, List, Optional
from transformer_lens import HookedTransformer, utils
from functools import partial
import einops
from dataclasses import dataclass
from transformer_lens.hook_points import HookPoint
import pickle
from jaxtyping import Int, Float
import torch as t
from torch import Tensor
Head = Tuple[int, int]

class HeadResults:
    data: Dict[Head, Tensor]
    def __init__(self, data=None):
        self.data = data or {} # ! bad practice to have default arguments be dicts

    def __getitem__(self, layer_and_head) -> Tensor:
        return self.data[layer_and_head].clone()
    
    def __setitem__(self, layer_and_head, value):
        self.data[layer_and_head] = value.clone()


class LayerResults:
    data: Dict[int, Any]
    def __init__(self, data=None):
        self.data = data or {}
    def __getitem__(self, layer: int) -> Any:
        return self.data[layer]
    def __setitem__(self, layer: int, value):
        self.data[layer] = value


@dataclass(frozen=False)
class LogitResults:
    zero_patched: HeadResults = HeadResults()
    mean_patched: HeadResults = HeadResults()
    zero_direct: HeadResults = HeadResults()
    mean_direct: HeadResults = HeadResults()


@dataclass(frozen=False)
class ModelResults:
    logits_orig: Tensor = t.empty(0)
    loss_orig: Tensor = t.empty(0)
    result: HeadResults = HeadResults()
    result_mean: HeadResults = HeadResults()
    pattern: HeadResults = HeadResults()
    v: HeadResults = HeadResults() # for value-weighted attn
    out_norm: HeadResults = HeadResults() # for value-weighted attn
    direct_effect: HeadResults = HeadResults()
    direct_effect_mean: HeadResults = HeadResults()
    resid_pre: HeadResults = HeadResults() # intermediate (for unembedding components)
    scale_attn: HeadResults = HeadResults() # intermediate (for unembedding components)
    unembedding_components: LayerResults = LayerResults()
    scale: Tensor = t.empty(0)
    logits: LogitResults = LogitResults()
    loss: LogitResults = LogitResults()

    def clear(self):
        # Empties all intermediate results which we don't need
        self.result = HeadResults()
        self.result_mean = HeadResults()
        self.resid_pre = HeadResults()
        self.v = HeadResults()

    def save(self, filename: str):
        # Saves self as pickle file
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def items(self):
        return self.__dict__.items()



def get_model_results(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_heads: List[Tuple[int, int]],
    use_cuda: bool = False,
) -> ModelResults:
    
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()

    current_device = str(next(iter(model.parameters())).device)
    if use_cuda and current_device == "cpu":
        model = model.cuda()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cpu()

    model_results = ModelResults()

    unembeddings: Float[Tensor, "batch seq d_model"] = model.W_U.T[toks]

    # Cache the head results and attention patterns, and final ln scale, and residual stream pre heads
    # (note, this is preferable to using names_filter argument for cache, because you can't pick specific heads)

    def cache_head_result(result: Float[Tensor, "batch seq n_heads d_model"], hook: HookPoint, head: int):
        model_results.result[hook.layer(), head] = result[:, :, head]
    
    def cache_head_pattern(pattern: Float[Tensor, "batch n_heads seq_Q seq_K"], hook: HookPoint, head: int):
        model_results.pattern[hook.layer(), head] = pattern[:, head]
    
    def cache_head_v(v: Float[Tensor, "batch seq n_heads d_head"], hook: HookPoint, head: int):
        model_results.v[hook.layer(), head] = v[:, :, head]
    
    def cache_scale(scale: Float[Tensor, "batch seq 1"], hook: HookPoint):
        model_results.scale = scale

    def cache_resid_pre(resid_pre: Float[Tensor, "batch seq d_model"], hook: HookPoint):
        model_results.resid_pre[hook.layer()] = resid_pre

    def cache_scale_attn(scale: Float[Tensor, "batch seq 1"], hook: HookPoint):
        model_results.scale_attn[hook.layer()] = scale
    

    all_layers = sorted(set([layer for layer, head in negative_heads]))
    for layer in all_layers:
        model.add_hook(utils.get_act_name("resid_pre", layer), cache_resid_pre)
        model.add_hook(utils.get_act_name("scale", layer, "ln1"), cache_scale_attn)
    for layer, head in negative_heads:
        model.add_hook(utils.get_act_name("result", layer), partial(cache_head_result, head=head))
        model.add_hook(utils.get_act_name("v", layer), partial(cache_head_v, head=head))
        model.add_hook(utils.get_act_name("pattern", layer), partial(cache_head_pattern, head=head))
    model.add_hook(utils.get_act_name("scale"), cache_scale)

    # Run the forward pass, to cache all values (and get logits)

    model_results.logits_orig, model_results.loss_orig = model(toks, return_type="both", loss_per_token=True)

    # Calculate the unembedding components stored in the residual stream, for each word in context

    for layer in all_layers:
        resid_pre_scaled: Float[Tensor, "batch seq d_model"] = model_results.resid_pre[layer] / model_results.scale_attn[layer]
        assert t.all((resid_pre_scaled.norm(dim=-1) - model.cfg.d_model ** 0.5) < 1e-4)

        seq_len = resid_pre_scaled.size(1)
        k = min(10, seq_len)
        q_indices = einops.repeat(t.arange(seq_len), "seq_Q -> seq_Q seq_K", seq_K=seq_len)
        k_indices = einops.repeat(t.arange(seq_len), "seq_K -> seq_Q seq_K", seq_Q=seq_len)
        causal_mask = (q_indices >= k_indices)

        q_tokens = einops.repeat(toks, "batch seq_Q -> batch seq_Q seq_K", seq_K=seq_len)
        k_tokens = einops.repeat(toks, "batch seq_K -> batch seq_Q seq_K", seq_Q=seq_len)
        causal_mask_rm_self = (q_indices >= k_indices) & (q_tokens != k_tokens)

        unembedding_components: Float[Tensor, "batch seq d_model"] = einops.einsum(
            resid_pre_scaled, unembeddings, # resid_pre_scaled[:, 1:], unembeddings[:, :-1],
            "batch seq_Q d_model, batch seq_K d_model -> batch seq_Q seq_K"
        )
        unembedding_components_avg = (unembedding_components * causal_mask).sum(dim=-1) / causal_mask.sum(dim=-1)
        unembedding_components_top10 = t.where(
            causal_mask, unembedding_components, t.full_like(unembedding_components, fill_value=-1e9)
        ).topk(k, dim=-1)
        
        unembedding_components_avg_rm_self = (unembedding_components * causal_mask_rm_self).sum(dim=-1) / causal_mask_rm_self.sum(dim=-1)
        unembedding_components_top10_rm_self = t.where(
            causal_mask_rm_self, unembedding_components, t.full_like(unembedding_components, fill_value=-1e9)
        ).topk(k, dim=-1)

        model_results.unembedding_components[layer] = {
            "avg": [unembedding_components_avg, unembedding_components_avg_rm_self],
            "top10": [unembedding_components_top10, unembedding_components_top10_rm_self],
        }

    # Get output norms for value-weighted attention

    for layer, head in negative_heads:
        out = einops.einsum(
            model_results.v[layer, head], model.W_O[layer, head],
            "batch seq d_head, d_head d_model -> batch seq d_model"
        )
        out_norm = einops.reduce(out.pow(2), "batch seq d_model -> batch seq", "sum").sqrt()
        model_results.out_norm[layer, head] = out_norm

    # Calculate the thing we'll be subbing in for mean ablation

    for layer, head in negative_heads:
        model_results.result_mean[layer, head] = einops.reduce(
            model_results.result[layer, head], 
            "batch seq d_model -> d_model", "mean"
        )

    # Now, use "result" to get the thing we'll eventually be adding to logits (i.e. scale it and map it through W_U)

    for layer, head in negative_heads:

        # TODO - is it more reasonable to patch in at the final value of residual stream instead of directly changing logits?
        model_results.direct_effect[layer, head] = einops.einsum(
            model_results.result[layer, head] / model_results.scale,
            model.W_U,
            "batch seq d_model, d_model d_vocab -> batch seq d_vocab"
        )
        model_results.direct_effect_mean[layer, head] = einops.reduce(
            model_results.direct_effect[layer, head],
            "batch seq d_vocab -> d_vocab",
            "mean"
        )

    # Two new forward passes: one with mean ablation, one with zero ablation. We only store logits from these

    def patch_head_result(
        result: Float[Tensor, "batch seq n_heads d_model"],
        hook: HookPoint,
        head: int,
        ablation_values: Optional[HeadResults] = None,
    ):
        if ablation_values is None:
            result[:, :, head] = t.zeros_like(result[:, :, head])
        else:
            result[:, :, head] = ablation_values[hook.layer(), head]
        return result

    for layer, head in negative_heads:
        model.add_hook(utils.get_act_name("result", layer), partial(patch_head_result, head=head))
        model_results.logits.zero_patched[layer, head] = model(toks, return_type="logits")
        model.add_hook(utils.get_act_name("result", layer), partial(patch_head_result, head=head, ablation_values=model_results.result_mean))
        model_results.logits.mean_patched[layer, head] = model(toks, return_type="logits")
    
    model_results.clear()

    # Now, the direct effects

    for layer, head in negative_heads:
        # Get the change in logits from removing the direct effect of the head
        model_results.logits.zero_direct[layer, head] = model_results.logits_orig - model_results.direct_effect[layer, head]
        # Get the change in logits from removing the direct effect of the head, and replacing with the mean effect
        model_results.logits.mean_direct[layer, head] = model_results.logits.zero_direct[layer, head] + model_results.direct_effect_mean[layer, head]

    # Calculate the loss for all of these
    for k in ["zero_patched", "mean_patched", "zero_direct", "mean_direct"]:
        setattr(model_results.loss, k, HeadResults({
            (layer, head): model.loss_fn(getattr(model_results.logits, k)[layer, head], toks, per_token=True)
            for layer, head in negative_heads
        }))

    if use_cuda and current_device == "cpu":
        model = model.cpu()
    elif (not use_cuda) and current_device != "cpu":
        model = model.cuda()

    return model_results