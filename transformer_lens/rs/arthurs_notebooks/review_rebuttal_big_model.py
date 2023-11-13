#%%

import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
import transformers
from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
import transformers
import datasets
import gc

#%% # Note 3 minute delay

model_name = "mistralai/Mistral-7B-v0.1"

#%%

model = transformer_lens.HookedTransformer.from_pretrained_no_processing(model_name)

#%%

if False:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name) # Hopefully works
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

#%%

print(model) # What do the modules look like? We'll try to fold the LN bias.

#%% # Takes 90 secs

model = model.to(torch.bfloat16) # Load into lower precision...
gc.collect()
torch.cuda.empty_cache()

#%% # Takes 10 seconds

model = model.cuda()
gc.collect()
torch.cuda.empty_cache()

# %%

# Sanity check that we can run forward passes 

tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
my_sentence = """Successorship behavior: the successor head pushes for the successor of a token in the context. We
say this behavior occurs when one of the top-5-attended tokens is in the successorship dataset, and
the correct next token is the successor of t.
Acronym behavior: the successor head pushes for an acronym of words in the context. We say this
behavior occurs when the correct next token is an acronym whose last letter corresponds to the first
letter of the top-1-attended token. (For example, if the successor head attends most strongly to the
token ‘Defense’, and the correct next token is ‘OSD’.)
Copying behavior: the successor head pushes for a previous token in the context. We say this
behavior occurs when the correct next token t has already occurred in the prompt, and token t is one
of the top-5-attended tokens.
Greater-than behavior: the successor head pushes for a token greater than a previous token in the
context. We say this behavior occurs when we do not observe successorship behavior, but when the
correct next token is still part of an ordinal sequence and has greater order than some top-5-attended
token (e.g. if the successor head attends to the token ‘first’ and the model predicts the token ‘third’.)"""
my_tokens = tokenizer.encode(my_sentence, return_tensors="pt").cuda() # First loss is really high but whatever, it's really low on average

#%%

my_logits = model(my_tokens).logits
my_logprobs = my_logits.log_softmax(dim=-1)
my_neglogprobs = -my_logprobs
my_loss = my_neglogprobs[0, torch.arange(my_neglogprobs.shape[1]-1), my_tokens[0][1:]]
print(my_loss.mean()) # Yeah pretty small

#%%

# Probably use the `streaming` option on this...
hf_dataset_name = "suolyer/pile_pile-cc"
# Dataset

#%%

hf_iter_dataset = iter(datasets.load_dataset(hf_dataset_name, streaming=True))

#%%

# %%

# From https://github.com/UlisseMini/activation_additions_hf/blob/cb418413ac5319d0ec698b84744257e7306a4f18/activation_additions/__init__.py#L15

from contextlib import contextmanager
from typing import Tuple, List, Callable, Optional
import torch as t
import torch.nn as nn


# types
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
Hook = Tuple[nn.Module, PreHookFn]
Hooks = List[Hook]


@contextmanager
def pre_hooks(hooks: Hooks):
    """
    Context manager to register pre-forward hooks on a list of modules. The hooks are removed when the context exits.
    """
    try:
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def get_blocks(model: nn.Module) -> nn.ModuleList:
    """
    Get the ModuleList containing the transformer blocks from a model.
    """
    def numel_(mod):
        return sum(p.numel() for p in mod.parameters())
    model_numel = numel_(model)
    canidates = [
        mod
        for mod in model.modules()
        if isinstance(mod, nn.ModuleList) and numel_(mod) > .5*model_numel
    ]
    assert len(canidates) == 1, f'Found {len(canidates)} ModuleLists with >50% of model params.'
    return canidates[0]


@contextmanager
def residual_stream(model: nn.Module, layers: Optional[List[int]] = None) -> List[t.Tensor]:
    """
    Context manager to store residual stream activations in the model at the specified layers.
    Alternatively "model(..., output_hidden_states=True)" can be used, this is more flexible though and works with model.generate().
    """

    stream = [None] * len(get_blocks(model))
    layers = layers or range(len(stream))
    def _make_hook(i):
        def _hook(_, inputs):
            # concat along the sequence dimension
            stream[i] = inputs[0] if stream[i] is None else t.cat([stream[i], inputs[0]], dim=1)
        return _hook

    hooks = [(layer, _make_hook(i)) for i, layer in enumerate(get_blocks(model)) if i in layers]
    with pre_hooks(hooks):
        yield stream


def _device(model):
    "Get the device of the first parameter of the model. Assumes all parameters are on the same device."
    return next(model.parameters()).device


def get_vectors(model: nn.Module, tokenizer, prompts: List[str], layer: int, padding=True):
    """
    Get the activations of the prompts at the specified layer. Used later for activation additions.
    """
    with residual_stream(model, layers=[layer]) as stream:
        inputs = tokenizer(prompts, return_tensors='pt', padding=padding)
        device = _device(model)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

    return stream[layer]


def get_diff_vector(model: nn.Module, tokenizer, prompt_add: str, prompt_sub: str, layer: int):
    """
    Get the difference vector between the activations of prompt_add and prompt_sub at the specified layer. 
    Convinience function for
        s = get_vectors(..., prompts)
        return s[0] - s[1]
    """
    stream = get_vectors(model, tokenizer, [prompt_add, prompt_sub], layer)
    return (stream[0] - stream[1]).unsqueeze(0)


def get_hook_fn(act_add: t.Tensor) -> PreHookFn:
    """
    Get a hook function that adds an activation addition vector.
    """

    def _hook(_: nn.Module, inputs: Tuple[t.Tensor]):
        resid_pre, = inputs
        if resid_pre.shape[1] == 1:
            return None # caching for new tokens in generate()

        # We only add to the prompt (first call), not the generated tokens.
        ppos, apos = resid_pre.shape[1], act_add.shape[1]
        assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

        # TODO: Make this a function-wrapper for flexibility.
        resid_pre[:, :apos, :] += act_add
        return resid_pre

    return _hook

#%%

# get_blocks(model)

get_vectors_output = get_vectors(
    model,
    tokenizer,
    ["The quick brown fox jumped over the lazy dog."],
    layer=0,
    padding=False, # Single token crap for now
)

# %%

with residual_stream(model, [0]) as f:
    pass

print(f)

# %%
