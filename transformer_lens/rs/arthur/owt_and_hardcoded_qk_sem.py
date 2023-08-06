#%%

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum2.generate_st_html.utils import (
    ST_HTML_PATH,
    parse_str_tok_for_printing,
)
from transformer_lens.rs.callum2.generate_st_html.model_results import (
    get_model_results,
)
from transformer_lens.rs.callum2.generate_st_html.generate_html_funcs import (
    generate_html_for_cspa_plots,
    generate_html_for_logit_plot,
    generate_html_for_DLA_plot,
    generate_4_html_plots,
    CSS,
)
from transformer_lens.rs.callum2.cspa.cspa_semantic_similarity import (
    get_equivalency_toks,
    get_related_words,
    concat_lists,
    is_token,
    get_list_with_no_repetitions,
    make_list_correct_length,
    create_full_semantic_similarity_dict,
)
from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project, get_effective_embedding_2
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
import argparse
clear_output()

# %%

"""
WARNING: Lots of copy+paste from ov_per_position.py
"""

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)

#%%

cspa_semantic_dict = pickle.load(open(ST_HTML_PATH.parent.parent / "cspa/cspa_semantic_dict_full.pkl", "rb"))

#%%

fpath = ST_HTML_PATH.parent.parent / "cspa/cspa_semantic_dict_full_token_idx_version.pkl"

if not fpath.exists():
    token_idx_version = {}
    succ = 0
    fails = 0

    for k in tqdm(cspa_semantic_dict):
        try:
            token_idx = model.to_single_token(k)
        except Exception:
            fails += 1
            continue
        assert token_idx not in token_idx_version

        token_idx_version[token_idx] = set()

        for lis in cspa_semantic_dict[k]:
            for tok in lis:
                try:
                    child_token_idx = model.to_single_token(tok)
                except Exception:
                    fails += 1
                    continue

                if child_token_idx not in token_idx_version[token_idx]:
                    token_idx_version[token_idx].add(child_token_idx)
                    succ += 1

    input("Sure you want to save this 8MB file?")
    torch.save(token_idx_version, fpath) # hopefully not too beefy

else:
    token_idx_version = torch.load(fpath)

#%%

MAX_SEQ_LEN = 512
BATCH_SIZE = 25
SEED = 1
batched_tokens, targets = get_filtered_webtext(model, batch_size=BATCH_SIZE, seed=SEED, device="cuda", max_seq_len=MAX_SEQ_LEN)
effective_embeddings = get_effective_embedding_2(model)
JSON_FNAME = "../arthur/json_data"
TOTAL_EFFECT_MIDS = True

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
LAYER_IDX, HEAD_IDX = NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"

# warnings.warn("Changed to scores for a diff comparison")
# attention_pattern_hook_name = get_act_name("attn_scores", NEGATIVE_LAYER_IDX)
ln_final_name = "ln_final.hook_scale"
attention_pattern_hook_name = get_act_name("pattern", NEGATIVE_LAYER_IDX)
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name==get_act_name("resid_pre", 1)
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.hook_resid_pre"
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_result"
    or name==attention_pattern_hook_name
    or name==get_act_name("v", LAYER_IDX)
    or name==ln_final_name
)
logits, cache = model.run_with_cache(
    batched_tokens,
    names_filter=names_filter1,
)
gc.collect()
torch.cuda.empty_cache()

# %%

original_end_state = cache[END_STATE_HOOK]

batched_tokens_loss = get_metric_from_end_state(
    model=model,
    end_state=original_end_state,
    targets=targets,
)

# %%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
assert head_output.shape == (BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model)
normalized_head_output = head_output / (head_output.var(dim=-1, keepdim=True) + model.cfg.eps)

#%%

mean_head_output = einops.reduce(head_output, "b s d -> d", reduction="mean")
normalized_mean_head_output = mean_head_output / (mean_head_output.var(dim=-1, keepdim=True) + model.cfg.eps)

#%%

mean_ablated_end_states = cache[get_act_name("resid_post", model.cfg.n_layers-1)] - head_output + einops.repeat(mean_head_output, "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN)
mean_ablated_loss = get_metric_from_end_state(
    model=model,
    end_state=mean_ablated_end_states,
    targets=targets,
)

#%%

def setter_hook(z, hook, setting_value, setter_head_idx=None):

    if setter_head_idx is not None:
        assert list(z.shape) == [BATCH_SIZE, MAX_SEQ_LEN, model.cfg.n_heads, model.cfg.d_model]
        z[:, :, setter_head_idx] = setting_value

    else: 
        if len(z.shape) == 3:
            assert len(z.shape) == 3 == len(setting_value.shape)
            assert list(z.shape[:2]) == [BATCH_SIZE, MAX_SEQ_LEN] == list(setting_value.shape[:2]), f"z.shape: {z.shape}, setting_value.shape: {setting_value.shape}, {[BATCH_SIZE, MAX_SEQ_LEN]}"
        elif len(z.shape) == 4: # blegh annoying hack
            if len(setting_value.shape) == 3:
                setting_value = einops.repeat(setting_value, "a b c -> a b n c", n=model.cfg.n_heads)
            assert list(z.shape) == list(setting_value.shape), f"z.shape: {z.shape}, setting_value.shape: {setting_value.shape}"

        z[:] = setting_value

    return z

# %%

if TOTAL_EFFECT_MIDS:
    
    # Get some different calculations of loss that come from just ablating the direct effect
    
    model.reset_hooks()
    model.add_hook(
        "blocks.10.attn.hook_result",
        partial(setter_hook, setting_value=einops.repeat(mean_head_output, "d -> batch seq d", seq=MAX_SEQ_LEN, batch=BATCH_SIZE).clone(), setter_head_idx=NEGATIVE_HEAD_IDX), 
    )
    total_effect_end_state = model.run_with_cache(
        batched_tokens,
        names_filter=lambda name: name == END_STATE_HOOK,
    )[1][END_STATE_HOOK]
    model.reset_hooks()

    total_effect_end_loss = get_metric_from_end_state(
        model=model,
        end_state=total_effect_end_state,
        targets=targets,
        frozen_ln_scale=cache[ln_final_name],
    )

#%%

loss_to_use = total_effect_end_loss if TOTAL_EFFECT_MIDS else mean_ablated_loss

max_importance_examples = sorted(
    [
        (
            batch_idx,
            seq_idx,
            (loss_to_use-batched_tokens_loss)[batch_idx, seq_idx].item(),
        )
        for batch_idx, seq_idx in itertools.product(
            range(BATCH_SIZE), range(MAX_SEQ_LEN)
        )
    ],
    key=lambda x: x[2],
    reverse=True,
)

# %%

# Get the top 5% of things by importance
all_top_5_percent = max_importance_examples[: len(max_importance_examples)//20]

np.random.seed(799)
np.random.shuffle(all_top_5_percent)

# warnings.warn("No shuffle!!! Instead sort by near coming")
# all_top_5_percent = sorted(
#     all_top_5_percent,
#     key=lambda x: x[1],
#     reverse=False,
# )

top_5_percent = all_top_5_percent[: BATCH_SIZE]
top5p_batch_indices = [x[0] for x in top_5_percent]
top5p_seq_indices = [x[1] for x in top_5_percent]

#%%

top5p_tokens = batched_tokens[top5p_batch_indices]
top5p_targets = torch.LongTensor([targets[top5p_batch_idx, top5p_seq_idx] for top5p_batch_idx, top5p_seq_idx in zip(top5p_batch_indices, top5p_seq_indices)])
top5p_end_states = original_end_state[top5p_batch_indices, top5p_seq_indices]
head_output = head_output[top5p_batch_indices, top5p_seq_indices]
top5p_loss_to_use = loss_to_use[top5p_batch_indices, top5p_seq_indices]

#%%

top5p_losses = batched_tokens_loss[top5p_batch_indices, top5p_seq_indices]

#%%

# 1. Make thing that calculates OV independently for each position
# 2. Check it agrees with normal stuff!
# 3. Get Neel's results : ) 
# 4. Generalize

W_E = model.W_E.clone()
W_U = model.W_U.clone()
W_O = model.W_O.clone()[LAYER_IDX, HEAD_IDX] # d_head d_model

#%%

resid_pre = cache[get_act_name("resid_pre", LAYER_IDX)]
head_pre = model.blocks[LAYER_IDX].ln1(resid_pre)
head_pre_normalized = head_pre / (head_pre.var(dim=-1, keepdim=True) + model.cfg.eps)
head_v = cache[get_act_name("v", LAYER_IDX)][:, :, HEAD_IDX, :]
head_pattern = cache[get_act_name("pattern", LAYER_IDX)][:, HEAD_IDX, :, :]

positionwise_z = einops.einsum(
    head_v,
    head_pattern,
    "batch key_pos d_head, \
    batch query_pos key_pos -> \
    batch query_pos key_pos d_head", # contributions from each source position
)

#%%

top5p_positionwise_z = positionwise_z[top5p_batch_indices, top5p_seq_indices]
del positionwise_z
gc.collect()
torch.cuda.empty_cache()

#%%

top5p_positionwise_out = einops.einsum(
    top5p_positionwise_z,
    W_O,
    "batch key_pos d_head, \
    d_head d_model -> \
    batch key_pos d_model",
)

#%%

top_unembeds_per_position = einops.einsum(
    top5p_positionwise_out,
    W_U,
    "batch key_pos d_model, \
    d_model d_vocab -> \
    batch key_pos d_vocab",
)
print(top_unembeds_per_position.norm())

#%%

total_unembed = einops.reduce(
    top_unembeds_per_position,
    "batch key_pos d_vocab -> batch d_vocab",
    reduction="sum",
)
print(total_unembed.norm())

#%%

average_unembed = einops.einsum(
    normalized_mean_head_output,
    W_U,
    "d_model, d_model d_vocab -> d_vocab",
)
print(average_unembed.norm())

#%%

logit_lens_pre_ten = einops.einsum(
    head_pre_normalized,
    W_U,
    "batch pos d_model, \
    d_model d_vocab -> \
    batch pos d_vocab",
)
print(logit_lens_pre_ten.norm())

#%%

logit_lens_pre_ten_probs = logit_lens_pre_ten.softmax(dim=-1)
logit_lens_pre_ten_probs_cpu = logit_lens_pre_ten_probs.cpu()

#%%

k_logit_lens = 50

logit_lens_top_pre_ten_probs = list(zip(*list(torch.topk(
    logit_lens_pre_ten_probs_cpu,
    dim = -1, 
    k = k_logit_lens,
))))

#%%

logit_lens_of_head = einops.einsum(
    normalized_head_output,
    W_U,
    "batch pos d_model, \
    d_model d_vocab -> \
    batch pos d_vocab",
)

#%%

logit_lens_head_bottom_ten = torch.topk(
    -logit_lens_of_head.cpu(),
    k=10,
    dim=-1,
)

#%%

ABS_MODE = True

def to_string(toks):
    s = model.to_string(toks)
    s = s.replace("\n", "\\n")
    return "|" + s

#%%

cpu_probs = logits.softmax(dim=-1).cpu()
top_probs = list(zip(*list(torch.topk(
    cpu_probs,
    dim = -1, 
    k = 10,
))))

#%%

for batch_idx in range(31): #range(len(top_unembeds_per_position)):
    assert top5p_seq_indices[batch_idx]+2 <= top_unembeds_per_position.shape[1], (top5p_seq_indices[batch_idx], top_unembeds_per_position.shape[1])
    the_logits = -top_unembeds_per_position[batch_idx][1:top5p_seq_indices[batch_idx]+2]
    if ABS_MODE:  # WAT
        the_logits = torch.abs(the_logits)
    max_logits = the_logits[:, 1:-1].max().item()
    my_obj = cv.logits.token_log_probs( # I am using this in a very cursed way: 
        top5p_tokens[batch_idx][:top5p_seq_indices[batch_idx]+1],
        the_logits - max_logits,
        to_string = to_string
    )

    print(f"True completion for {SEED=} {top5p_batch_indices[batch_idx]=} {top5p_seq_indices[batch_idx]=}:"+model.to_string(top5p_tokens[batch_idx][top5p_seq_indices[batch_idx]+1]))

    print(
        "\n10.7 Attentions\n",
    )
    top_attention_weight_index = torch.topk(
        head_pattern[top5p_batch_indices[batch_idx], top5p_seq_indices[batch_idx], 1:top5p_seq_indices[batch_idx]+1],
        k=1, 
    ).indices.item() + 1

    top_attention_weight_token = top5p_tokens[batch_idx][top_attention_weight_index].item()

    topk_model_predictions = logit_lens_top_pre_ten_probs[top5p_batch_indices[batch_idx]][1][top5p_seq_indices[batch_idx]]
    project_onto_predictions, _ = project(
        einops.repeat(
            head_pre_normalized[top5p_batch_indices[batch_idx], top5p_seq_indices[batch_idx], :],
            "d_model -> k_projs d_model",
            k_projs=k_logit_lens,
        ).clone(),
        model.W_U.T[topk_model_predictions],
    )

    top_attention_scores = dot_with_query(
        unnormalized_queries=project_onto_predictions,
        unnormalized_keys=einops.repeat(head_pre_normalized[top5p_batch_indices[batch_idx], top_attention_weight_index, :], "d_model -> k_projs d_model", k_projs=k_logit_lens).clone(),
        model=model,
        layer_idx=LAYER_IDX,
        head_idx = HEAD_IDX,
        add_key_bias = True,
        add_query_bias=False,
        normalize_keys = False,
        normalize_queries = False,
    )

    try:
        cur_string = model.to_single_str_token(top_attention_weight_token)

        assert top_attention_weight_token in token_idx_version[top_attention_weight_token]

        # if top_attention_weight_token in topk_model_predictions:

        is_sem_sim = [str(topk_model_prediction.item() in token_idx_version[top_attention_weight_token]) for topk_model_prediction in topk_model_predictions]
        if top_attention_weight_token in topk_model_predictions:
            is_sem_sim[topk_model_predictions.tolist().index(top_attention_weight_token)] = "Exact same token"

        px.bar(
            x = [f"{i}: `{model.to_single_str_token(topk_model_prediction.item())}`" for i, topk_model_prediction in enumerate(topk_model_predictions)],
            y = top_attention_scores.tolist(),
            color=is_sem_sim,
            labels = {"x": "Model prediction", "y": "Attention score", "color": "Semantically similar?"},
            title = f"10.7 Attention Scores for `{cur_string}`",
        ).show()

    except Exception as e:
        print(e)
        print("Couldn't find token in semantic dict???")
        continue

    print("\nTop model predictions before 10.7:")
    cur_ten_probs = logit_lens_top_pre_ten_probs[top5p_batch_indices[batch_idx]]
    print(
        list(zip(cur_ten_probs[0][top5p_seq_indices[batch_idx]].tolist(), model.to_str_tokens(cur_ten_probs[1][top5p_seq_indices[batch_idx]]), cur_ten_probs[1][top5p_seq_indices[batch_idx]].tolist(), strict=True))
    )

    print("\nTop negative logit changes from 10.7:\n", model.to_str_tokens(logit_lens_head_bottom_ten.indices[top5p_batch_indices[batch_idx], top5p_seq_indices[batch_idx]])) # TODO make this show logits too?

    print("\nModel's top probs:")
    cur_top_probs = top_probs[top5p_batch_indices[batch_idx]]

    print(
        list(zip(cur_top_probs[0][top5p_seq_indices[batch_idx]].tolist(), model.to_str_tokens(cur_top_probs[1][top5p_seq_indices[batch_idx]]), cur_top_probs[1][top5p_seq_indices[batch_idx]].tolist(), strict=True))
    )

    display(my_obj)

# %%
