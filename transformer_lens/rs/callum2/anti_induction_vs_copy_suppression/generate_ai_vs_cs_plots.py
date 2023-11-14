#%%
#!/usr/bin/env python
# coding: utf-8

# # Generate AI vs CS plots
# 
# Open question - how much of anti-induction is actually just copy-suppression?
# 
# We're answering this question by doing a large scatter plot of two metrics.
# 
# On the x-axis is **copy-suppression scores on the IOI distribution**. This is calculated as follows:
# 
# * For the given attention head, take the result vector being moved from the IO token to the end token.
# * Measure its direct logit attribution in the direction of the IO token.
# 
# This should be very positive for copy heads, and very negative for our negative heads.
# 
# On the y-axis is **anti-induction scores on the IOI distribution**. This is calculated as follows:
# 
# * Input a random repeating sequence (i.e. the model's BOS token, followed by 2 copies of the same random sequence concatenated together).
# * Measure the model's direct logit attribution on the correct token.
# 
# This should be positive for induction heads, and negative for anti-induction heads.
# 
# ## How could this be improved?
# 
# The anti-induction metric is pretty clear and obvious. I'm not quite as happy with the IOI metric, because this is just one of the many cases where negative behaviour is displayed. However, it seems like a pretty crisp example.
# 
# Other possible ideas:
# 
# * Run the model on OWT (but this might take a long time!).
# * Take classic sentences, like the "breaking the pattern" example about picking up "the third and final box".
#     * However, an issue with this is that this is kinda anti-induction.
# * Measure directly from the weights - some combination of "average log self-attention rank" and "average log self-suppression rank" for non-function words.

# In[1]:


import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
from transformer_lens.cautils.notebook import *
t.set_grad_enabled(False)

from transformer_lens.rs.callum2.utils import (
    create_title_and_subtitles,
    parse_str,
    parse_str_tok_for_printing,
    parse_str_toks_for_printing,
    topk_of_Nd_tensor,
    ST_HTML_PATH,
    project,
)

from transformer_lens.rs.callum2.generate_st_html.model_results import (
    FUNCTION_STR_TOKS,
)

clear_output()


# # Paper figures

# In[2]:


media_path = Path("/home/ubuntu/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media/anti_induction") if os.path.exists("/home/ubuntu") else Path("/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media/anti_induction")
SCORES_DICT = pickle.load(open(media_path / "scores_dict.pkl", "rb"))
SCORES_DICT = {k: v for k, v in SCORES_DICT.items() if "opt" not in k}
MODEL_NAMES = sorted(SCORES_DICT.keys())


# In[3]:


param_sizes = {
    "gpt2": "85M",
    "gpt2-medium": "302M",
    "gpt2-large": "708M",
    "gpt2-xl": "1.5B",
    "distillgpt2": "42M",
    "opt-125m": "85M",
    "opt-1.3b": "1.2B",
    "opt-2.7b": "2.5B",
    "opt-6.7b": "6.4B",
    "opt-13b": "13B",
    "opt-30b": "30B",
    "opt-66b": "65B",
    "gpt-neo-125m": "85M",
    "gpt-neo-1.3b": "1.2B",
    "gpt-neo-2.7b": "2.5B",
    "gpt-neo-1.3B": "1.2B",
    "gpt-neo-2.7B": "2.5B",
    "gpt-j-6B": "5.6B",
    "gpt-neox-20b": "20B",
    "stanford-gpt2-small-a": "85M",
    "stanford-gpt2-small-b": "85M",
    "stanford-gpt2-small-c": "85M",
    "stanford-gpt2-small-d": "85M",
    "stanford-gpt2-small-e": "85M",
    "stanford-gpt2-medium-a": "302M",
    "stanford-gpt2-medium-b": "302M",
    "stanford-gpt2-medium-c": "302M",
    "stanford-gpt2-medium-d": "302M",
    "stanford-gpt2-medium-e": "302M",
    "pythia-70m": "19M",
    "pythia-160m": "85M",
    "pythia-410m": "302M",
    "pythia-1b": "5M",
    "pythia-1.4b": "1.2B",
    "pythia-2.8b": "2.5B",
    "pythia-6.9b": "6.4B",
    "pythia-12b": "11B",
    "pythia-70m-deduped": "19M",
    "pythia-160m-deduped": "85M",
    "pythia-410m-deduped": "302M",
    "pythia-1b-deduped": "805M",
    "pythia-1.4b-deduped": "1.2B",
    "pythia-2.8b-deduped": "2.5B",
    "pythia-6.9b-deduped": "6.4B",
    "pythia-12b-deduped": "11B",
    "solu-4l": "13M",
    "solu-6l": "42M",
    "solu-8l": "101M",
    "solu-10l": "197M",
    "solu-12l": "340M",
    "solu-4l-pile": "13M",
    "solu-6l-pile": "42M",
    "solu-8l-pile": "101M",
    "solu-10l-pile": "197M",
    "solu-12l-pile": "340M",
    "solu-1l": "3.1M",
    "solu-2l": "6.3M",
    "solu-3l": "9.4M",
    "solu-4l": "13M",
    "solu-6l": "42M",
    "solu-8l": "101M",
    "solu-10l": "197M",
    "solu-12l": "340M",
    "gelu-1l": "3.1M",
    "gelu-2l": "6.3M",
    "gelu-3l": "9.4M",
    "gelu-4l": "13M",
    "attn-only-1l": "1.0M",
    "attn-only-2l": "2.1M",
    "attn-only-3l": "3.1M",
    "attn-only-4l": "4.2M",
    "attn-only-2l-demo": "2.1M",
    "solu-1l-wiki": "3.1M",
    "solu-4l-wiki": "13M",
    "llama-7b-hf": "7B",
    "mistralai/Mistral-7B-v0.1": "7B",
}

def get_size(model_name):
    size_str = param_sizes[model_name]
    if size_str.endswith("M"):
        size = int(1e6 * float(size_str[:-1]))
    elif size_str.endswith("B"):
        size = int(1e9 * float(size_str[:-1]))
    else:
        raise Exception
    return size

def get_model_class(model_name):
    for x in ["GPT", "OPT", "Pythia", "SoLU", "GELU", "Other", "Llama", "Mistral"]:
        if x.lower() in model_name.lower():
            return x
    else:
        return "Other"

# In[4]:


def plot_all_results(
    pospos=False,
    showtext=False,
    fraction=False,
    categories="all",
):
    results_copy_suppression_ioi = []
    results_anti_induction = []
    results_copy_suppression_norm = []
    model_names = []
    head_names = []
    fraction_list = []
    num_params = []
    model_classes = []

    for model_name in MODEL_NAMES:
        if ("opt" not in model_name.lower()) and ("gelu" not in model_name.lower()):
            model_scores = SCORES_DICT[model_name]
            for layer in range(model_scores.size(1)):
                for head in range(model_scores.size(2)):
                    results_copy_suppression_ioi.append(-model_scores[0, layer, head].item())
                    results_anti_induction.append(-model_scores[1, layer, head].item())
                    if model_scores.shape[0] == 3:
                        results_copy_suppression_norm.append(-model_scores[2, layer, head].item())
                    else:
                        results_copy_suppression_norm.append(np.nan)
                    model_names.append(model_name)
                    head_names.append(f"{layer}.{head}")
                    fraction_list.append((layer + 1) / model_scores.size(1))
                    num_params.append(get_size(model_name))
                    model_classes.append(get_model_class(model_name))

    df = pd.DataFrame({
        "results_cs_ioi": results_copy_suppression_ioi,
        "results_cs_norm": results_copy_suppression_norm,
        "results_ai_rand": results_anti_induction,
        "model_name": model_names,
        "model_class": model_classes,
        "Head name": head_names,
        "head_and_model_names": [f"{model_names[i]} [{head_names[i]}]" for i in range(len(model_names))],
        "fraction_list": fraction_list,
        "num_params": num_params,
    })

    # if cs_metric == "norm":
    #     df = df[df["results_cs_norm"] != np.nan]
    #     x = "results_cs_norm"
    # else:
    x = "results_cs_ioi"

    if pospos:
        is_pos = [i for i, (x, y) in enumerate(zip(results_copy_suppression_ioi, results_anti_induction)) if x > 0 and y > 0]
        df = df.iloc[is_pos]

    if categories.lower() == "none":
        df = df[df["model_name"] == ""] # filter everything out
    elif categories.lower() != "all":
        df = df[[categories.lower() in name for name in df["model_name"]]]

    fig = px.scatter(
        df,
        x=x, 
        y="results_ai_rand", 
        color='model_class' if not(fraction) else "fraction_list", 
        hover_name="head_and_model_names",
        hover_data={
            "Copy Suppression": [f"<b>{x:.3f}</b>" for x in df[x]],
            "Anti-Induction": [f"<b>{x:.3f}</b>" for x in df["results_ai_rand"]],
            "model_name": False,
            x: False,
            "results_ai_rand": False,
        },
        # text="head_and_model_names" if showtext else None,
        title="Anti-Induction Scores (repeated random tokens) vs Copy-Suppression Scores (IOI)",
        labels={
            x: "Copy-Suppression Score",
            "results_ai_rand": "Anti-Induction Score",
            "model_class": "Model Class",
        },
        height=650,
        width=1000,
        color_continuous_scale=px.colors.sequential.Rainbow if fraction else None,
        template="simple_white",
    )
    # fig.update_layout(legend_title_font_size=18)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_traces(textposition='top center')
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(
        yaxis_range=[-0.5, 1.5],
        xaxis_range=[-0.5, 2.8],
    )
    fig.add_annotation(
        x=2.382,
        y=1.276,
        text="GPT2-Small, L10H7",
        showarrow=True,
        font_size=14,
    )

    # # Now we add legend groups
    # for trace in fig.data:
    #     for group_substr in ["GPT", "OPT", "Pythia", "SoLU", "GELU", "Other"]:
    #         if group_substr.lower() in trace.name.lower():
    #             break
    #     trace.legendgroup = group_substr
    #     trace.legendgrouptitle = {'text': f"<br><span style='font-size:16px'>{'='*8} {group_substr} {'='*8}</span>"}

    # fig.update_layout(paper_bgcolor='rgba(255,244,214,0.5)', plot_bgcolor='rgba(255,244,214,0.5)')
    return fig


# In[5]:


fig = plot_all_results(
    pospos=False,
    showtext=False,
    fraction=False,
    categories="all",
)
fig.update_layout(height=550, width=800)

MEDIA_PATH = Path(r"/home/ubuntu/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media") if os.path.exists("/home/ubuntu") else Path(r"/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media")
from plotly.utils import PlotlyJSONEncoder
import json
with open(MEDIA_PATH / "ai_vs_cs.json", 'w') as f:
    f.write(json.dumps(fig, cls=PlotlyJSONEncoder))
# fig.write_image(media_path / "ai_vs_cs.pdf")

fig.show()


# In[6]:


fig = plot_all_results(
    pospos=False,
    showtext=False,
    fraction=False,
    categories="all",
)

fig.show()
fig.write_image(media_path / "ai_vs_cs.pdf")


# # Setup figures

# In[7]:


device = "cuda:0"

if False:
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device=device,
        # refactor_factored_attn_matrices=True,
    )

else:
    # model_name = "huggyllama/llama-7b"
    # model_name = "mistralai/Mistral-7B-v0.1"
    model_name = "distillgpt2"

    # model_name = "gpt2-small"
    import transformers
    from transformer_lens import HookedTransformer # Some TODOs; script this so we can sweep all the darn models. Also, check that code isn't bugged; ensure things work normally for GPT-2 Small
    import torch
    # from_pretrained version 4 minutes
    long_path = "/workspace/cache/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658/"
    if "llama" in model_name.lower() or "mistral" in model_name.lower():
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(long_path) if "mistral" in model_name.lower() and os.path.exists(long_path) else transformers.AutoModelForCausalLM.from_pretrained(model_name)
        print("...")
        model = hf_model.to(torch.bfloat16)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        print("Got tokenizer...")
        model = HookedTransformer.from_pretrained_no_processing("llama-7b" if "llama" in model_name.lower() else model_name, hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=hf_tokenizer, dtype=torch.bfloat16)
        print("Got model in TL...")
    else:
        model = HookedTransformer.from_pretrained(model_name, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)

    torch.set_grad_enabled(False)
    model = model.to(torch.bfloat16)
    model = model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

model.set_use_attn_result(False)
clear_output()


# In[9]:


def get_copy_suppression_scores_ioi(model: HookedTransformer, N: int):

    all_results = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device, dtype=t.float)

    ioi_dataset, ioi_cache = generate_data_and_caches(
        N,
        model,
        seed=42,
        prepend_bos=True,
        only_ioi=True,
        symmetric=True,
        names_filter=lambda name: any([name.endswith(x) for x in ["scale", "v", "pattern"]])
    )

    io_unembeddings = model.W_U.T[ioi_dataset.io_tokenIDs] # (batch, d_model)

    scale = ioi_cache["scale"] # (batch, seq, 1)
    scale = scale[range(N), ioi_dataset.word_idx["end"]] # (batch, 1)

    for layer in range(model.cfg.n_layers):
        v = ioi_cache["v", layer] # (batch, seq, n_heads, d_head)

        v_io = v[range(N), ioi_dataset.word_idx["IO"]] # (batch, n_heads, d_head)

        # Get result (before attn patterns)
        result_io = einops.einsum(
            v_io, model.W_O[layer],
            "batch n_heads d_head, n_heads d_head d_model -> batch n_heads d_model"
        )

        # Get result moved to `end` token (after attn patterns)
        patterns = ioi_cache["pattern", layer] # (batch, n_heads, seqQ, seqK)
        patterns_end_to_io = patterns[range(N), :, ioi_dataset.word_idx["end"], ioi_dataset.word_idx["IO"]] # (batch, n_heads)
        result_io_to_end = einops.einsum(
            result_io, patterns_end_to_io,
            "batch n_heads d_model, batch n_heads -> batch n_heads d_model"
        ).to(result_io.dtype)# SOmetimes attention mixed precision

        # Finally, get attribution (which includes effect of layernorm)
        dla = einops.einsum(
            result_io_to_end, io_unembeddings,
            "batch n_heads d_model, batch d_model -> batch n_heads"
        ) / scale
        dla = einops.reduce(dla, "batch n_heads -> n_heads", "mean")
        
        all_results[layer] = dla

    return all_results

# In[10]:


def get_anti_induction_scores(model: HookedTransformer, N: int, seq_len: int = 30):

    tokens_to_repeat = t.randint(0, model.cfg.d_vocab, (N, seq_len), device=device)
    bos_tokens = t.full((N, 1), model.tokenizer.bos_token_id, device=device, dtype=t.long)
    tokens = t.concat([bos_tokens, tokens_to_repeat, tokens_to_repeat], dim=1)
    assert tokens.shape == (N, 2*seq_len+1)
        
    all_results = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device, dtype=t.float)

    _, cache = model.run_with_cache(
        tokens,
        return_type = None,
        names_filter = lambda name: any([name.endswith(x) for x in ["scale", "v", "pattern"]])
    )

    rep_unembeddings = model.W_U.T[tokens_to_repeat[:, 1:]] # (batch, rep_seq_pos, d_model)

    batch_indices = einops.repeat(t.arange(N, device=device), "batch -> batch seq", seq=seq_len-1)
    dest_indices = einops.repeat(t.arange(seq_len+1, 2*seq_len, device=device), "seq -> batch seq", batch=N)
    src_indices = einops.repeat(t.arange(2, seq_len+1, device=device), "seq -> batch seq", batch=N)

    scale = cache["scale"] # (batch, seq, 1)
    scale = scale[batch_indices, dest_indices] # (batch, rep_seq_pos, 1)

    for layer in range(model.cfg.n_layers):
        v = cache["v", layer] # (batch, seq, n_heads, d_head)

        v_io = v[batch_indices, src_indices] # (batch, rep_seq_pos, n_heads, d_head)

        # Get result (before attn patterns)
        result_io = einops.einsum(
            v_io, model.W_O[layer],
            "batch rep_seq_pos n_heads d_head, n_heads d_head d_model -> batch rep_seq_pos n_heads d_model"
        )

        # Get result moved to dest tokens (after attn patterns)
        patterns = cache["pattern", layer] # (batch, n_heads, seqQ, seqK)
        patterns_end_to_io = patterns[batch_indices, :, dest_indices, src_indices] # (batch, rep_seq_pos, n_heads)
        result_io_to_end = einops.einsum(
            result_io, patterns_end_to_io,
            "batch rep_seq_pos n_heads d_model, batch rep_seq_pos n_heads -> batch rep_seq_pos n_heads d_model"
        ).to(result_io.dtype)

        # Finally, get attribution (which includes effect of layernorm)
        dla = einops.einsum(
            result_io_to_end, rep_unembeddings,
            "batch rep_seq_pos n_heads d_model, batch rep_seq_pos d_model -> batch rep_seq_pos n_heads"
        ) / scale
        dla = einops.reduce(dla, "batch rep_seq_pos n_heads -> n_heads", "mean")
        
        all_results[layer] = dla

    return all_results


# In[11]:


from transformer_lens import FactoredMatrix


def custom_effective_embedding(model: HookedTransformer, only_mlps: bool = False) -> Float[Tensor, "d_vocab d_model"]:

    resid = model.W_E.unsqueeze(0)

    pre_attention = model.blocks[0].ln1(resid)
    attn_out = einops.einsum(
        pre_attention, 
        model.W_V[0],
        model.W_O[0],
        "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
    )
    resid_mid = attn_out + resid
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    resid = resid_mid + mlp_out

    if only_mlps:
        W_EE0 = mlp_out.squeeze()
        return W_EE0
    else:
        W_EE = resid.squeeze()
        return W_EE


def get_weight_scores(model: HookedTransformer, N: int = 100, tied_embeddings: bool = False):
    '''
    Returns the mean log(rank+1) (from sampling) of two things:

        OV circuit metric: how much does a word suppress itself?
        QK circuit metric: how much does a word attend to itself?

    For instance, with head 10.7, most of the values will be 1 (or very small) because words suppress
    themselves & attend to themselves. So average log rank will be very close to zero. But for copying
    heads, most of the values will be pretty large, so the average log rank will be much larger.

    Since log has a much higher gradient at smaller values, this will single out super negative heads.
    Heads which are "middle neg/pos" or "highly positive" won't be very distinguishable.

    The result has shape (3, n_layers, n_heads), where:
        [0] -> OV scores (filtered for neg head results)
        [1] -> QK scores (filtered for neg head results)
        [2] -> The baseline-subtracted product
    '''
    W_U = model.W_U
    W_E = model.W_E if not(tied_embeddings) else custom_effective_embedding(model, only_mlps=True)

    results = t.zeros((2, model.cfg.n_layers, model.cfg.n_heads))

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):

            full_OV_matrix = FactoredMatrix(
                W_E @ model.W_V[layer][head],
                model.W_O[layer][head] @ W_U
            )
            full_QK_matrix = FactoredMatrix(
                W_E @ model.W_K[layer][head],
                model.W_Q[layer][head].T @ W_U
            )

            # Are the diagonal values the most negative (i.e. self-suppression)?
            random_indices = t.randint(0, model.cfg.d_vocab, (N,), device=device)
            OV_slice = full_OV_matrix.A[random_indices, :] @ full_OV_matrix.B
            OV_diag_values = OV_slice[range(N), random_indices].unsqueeze(1)
            OV_ranks = (OV_diag_values > OV_slice).sum(dim=1).float()
            OV_avg_log_rank = t.log(OV_ranks + 1).mean().item()

            # Are the diagonal values the most positive (i.e. prediction-attention)?
            random_indices = t.randint(0, model.cfg.d_vocab, (N,), device=device)
            QK_slice = full_QK_matrix.A @ full_QK_matrix.B[:, random_indices]
            QK_diag_values = QK_slice[random_indices, range(N)].unsqueeze(0)
            QK_ranks = (QK_diag_values < QK_slice).sum(dim=0).float()
            QK_avg_log_rank = t.log(QK_ranks + 1).mean().item()

            results[0, layer, head] = OV_avg_log_rank
            results[1, layer, head] = QK_avg_log_rank

    zero_point = t.tensor(model.cfg.d_vocab).log().item() - 1

    results = t.stack([
        results[0] - zero_point,
        results[1] - zero_point,
        (results[0] - zero_point) * (results[1] - zero_point),
    ])
    results[:2] *= (results[:2] < 0)
    results[2] *= -1 * (results[0] < 0) * (results[1] < 0)
    results[2] *= (results[:2].abs().max() / results[2].abs().max())

    return results


# In[12]:


BATCH_SIZE = 40 # 91 for scatter, 51 for viz
SEQ_LEN = 50 # 100 for scatter, 61 for viz (no more, cause attn)

def process_webtext_1(
    seed: int = 6,
    batch_size: int = BATCH_SIZE,
):
    DATA_STR = get_webtext(seed=seed)[:batch_size]
    DATA_STR = [parse_str(s) for s in DATA_STR]
    clear_output()
    return DATA_STR

def process_webtext_2(
    model: HookedTransformer,
    DATA_STR: List[str],
    seq_len: int = SEQ_LEN,
    verbose: bool = False,
) -> Int[Tensor, "batch seq"]:
    DATA_TOKS = model.to_tokens(DATA_STR)

    if seq_len < 1024:
        DATA_TOKS = DATA_TOKS[:, :seq_len]

    if verbose:
        print(f"Shape = {DATA_TOKS.shape}\n")
        # DATA_STR_TOKS = model.to_str_tokens(DATA_STR)
        # DATA_STR_TOKS = [str_toks[:seq_len] for str_toks in DATA_STR_TOKS]
        # DATA_STR_TOKS_PARSED = list(map(parse_str_toks_for_printing, DATA_STR_TOKS))
        # print("First prompt:\n" + "".join(DATA_STR_TOKS[0]))

    return DATA_TOKS.to(device)


DATA_STR = process_webtext_1()
# DATA_TOKS = process_webtext_2(gpt2, DATA_STR)
# BATCH_SIZE, SEQ_LEN = DATA_TOKS.shape


# In[13]:


def get_in_vivo_copy_suppression_scores(model: HookedTransformer, DATA_STR: Int[Tensor, "batch seq"]):
    '''
    This is the current preferred way of getting copy suppression scores for an attention head. The methodology is as follows:

    For each destination token, we...
        1. Find the source token S s.t. the unembedding of S is most present in the residual stream (i.e. logit lens), excluding any S = function words
        2. Take the result vector from that source token (multiplied by its attention probability)
        3. Project it onto the unembedding vector of the source token S
        4. Find the magnitude of the result

    Why is this a good CS metric? Because it...

        > Requires the QK circuit to exhibit prediction-attention (i.e. attend back to the tokens it is predicting)
        > Requires the OV circuit to exhibit copy-suppression (i.e. suppress the prediction of the source token)
        > Will be very small for any heads which don't strongly exhibit copy-suppression or copy-amplification (because we're 
          multiplying by attn and projecting onto a single direction)

    Why might this not be a good CS metric? Because it...

        > Will only pick up on "pure copy-suppression" rather than "fuzzy copy-suppression"
            > But that's okay, because pure CS is still a good fraction of what the head does, and so we should still be able to identify the neg heads

    What might be a better metric?

        Filter for high attention probabilities (e.g. 10x more than 1/seqQ), then only keep that result vector if the corresonding unembedding is present
        on the query-side. Not sure if this is better.
    '''
    toks = process_webtext_2(model, DATA_STR, SEQ_LEN)
    batch_size, seq_len = toks.shape

    _, cache = model.run_with_cache(
        toks,
        return_type = None,
        names_filter = lambda name: any(name.endswith(x) for x in ["pattern", "v", "resid_post", "scale"])
    )

    FUNCTION_TOKS = t.concat([
        model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze().to(device),
        t.tensor([model.tokenizer.bos_token_id]).to(device)
    ])

    results = t.zeros((model.cfg.n_layers, model.cfg.n_heads))

    # Get all source token unembeddings
    unembeddings_for_src_tokens: Float[Tensor, "batch seqK"] = model.W_U.T[toks]
    # Get useful indices
    batch_idx = einops.repeat(t.arange(batch_size), "b -> b sQ", sQ=seq_len)
    seqQ_idx = einops.repeat(t.arange(seq_len), "sQ -> b sQ sK", b=batch_size, sK=seq_len)
    seqK_idx = einops.repeat(t.arange(seq_len), "sK -> b sQ sK", b=batch_size, sQ=seq_len)

    for layer in range(model.cfg.n_layers):

        # Get everything we need from the cache
        resid_post_scaled: Float[Tensor, "batch seqQ d_model"] = cache["resid_post", layer] / cache["scale"]
        v: Float[Tensor, "batch seqK head d_head"] = cache["v", layer]
        pattern: Float[Tensor, "batch head seqQ seqK"] = cache["pattern", layer]

        # Get logit lens for src tokens, at each dest token
        logit_lens: Float[Tensor, "batch seqQ seqK"] = einops.einsum(
            resid_post_scaled, unembeddings_for_src_tokens,
            "batch seqQ d_model, batch seqK d_model -> batch seqQ seqK"
        )
        # Apply causal mask
        logit_lens = t.where(seqQ_idx >= seqK_idx, logit_lens, -1e9)
        # best_src_tok_seqpos[i, j] = the seqpos of the source token we're moving attn from, with destination token as (i, j)
        best_src_tok_seqpos: Int[Tensor, "batch seqQ"] = logit_lens.argmax(-1)
        best_src_toks: Int[Tensor, "batch seqQ"] = toks[batch_idx, best_src_tok_seqpos]

        # Get the actual things we're moving

        v_src: Float[Tensor, "batch seqQ head d_head"] = v[batch_idx, best_src_tok_seqpos]
        pattern_src_dest: Float[Tensor, "batch seqQ head"] = pattern[batch_idx, :, seqQ_idx[..., 0], best_src_tok_seqpos]

        # Do the projections & attention scaling
        result_src: Float[Tensor, "batch seqQ head d_model"] = einops.einsum(
            v_src, model.W_O[layer],
            "batch seqQ head d_head, head d_head d_model -> batch seqQ head d_model"
        )
        result_src_projections: Float[Tensor, "batch seqQ head"] = einops.einsum(
            result_src, unembeddings_for_src_tokens[batch_idx, best_src_tok_seqpos],
            "batch seqQ head d_model, batch seqQ d_model -> batch seqQ head"
        ) / cache["scale"]
        result_dest_projections: Float[Tensor, "batch seqQ head"] = result_src_projections * pattern_src_dest

        # Get a filter for where the source token was a function word (we don't include these)
        best_src_toks_are_fn_words: Bool[Tensor, "batch seqQ 1"] = (best_src_toks[:, :, None] == FUNCTION_TOKS[None, None, :]).any(dim=-1, keepdim=True)

        # Get the results for all the non-fn words
        mean_result_dest_projections = (result_dest_projections * ~best_src_toks_are_fn_words).sum(dim=(0, 1)) / (~best_src_toks_are_fn_words).sum(dim=(0, 1))
        results[layer] = mean_result_dest_projections

    return results


# In[14]:


def get_in_vivo_copy_suppression_scores_2(model: HookedTransformer, DATA_STR: Int[Tensor, "batch seq"]):
    '''
    Same as the other one, except rather than picking the top source token, it picks the top token over all 50k words, and sets the result to zero
    if that top token isn't in context. This is a lot more strict, and hopefully a lot more sparse.
    '''
    toks = process_webtext_2(model, DATA_STR, SEQ_LEN)
    batch_size, seq_len = toks.shape

    _, cache = model.run_with_cache(
        toks,
        return_type = None,
        names_filter = lambda name: any(name.endswith(x) for x in ["pattern", "v", "resid_pre", "scale"])
    )

    FUNCTION_TOKS = t.concat([
        model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze().to(device),
        t.tensor([model.tokenizer.bos_token_id]).to(device)
    ])

    results = t.zeros((model.cfg.n_layers, model.cfg.n_heads))

    for layer in range(model.cfg.n_layers):

        # Get everything we need from the cache
        resid_post_scaled: Float[Tensor, "batch seqQ d_model"] = cache["resid_pre", layer] / cache["scale"]
        v: Float[Tensor, "batch seqK head d_head"] = cache["v", layer]
        pattern: Float[Tensor, "batch head seqQ seqK"] = cache["pattern", layer]

        # Get logit lens for all tokens, at each dest token
        # Get the (batch, seqQ) indices of everywhere that the top token isn't a function word & is in context
        logit_lens: Float[Tensor, "batch seqQ d_vocab"] = resid_post_scaled @ model.W_U
        top_predicted_token: Int[Tensor, "batch seqQ"] = logit_lens.argmax(-1)
        top_predicted_token_is_non_fn_word: Bool[Tensor, "batch seqQ"] = (top_predicted_token[:, :, None] != FUNCTION_TOKS[None, :]).all(dim=-1)
        top_predicted_token_rep = einops.repeat(top_predicted_token, "batch seqQ -> batch seqQ seqK", seqK=seq_len)
        toks_rep = einops.repeat(toks, "batch seqK -> batch seqQ seqK", seqQ=seq_len)
        toks_rep = t.where(t.tril(t.ones((seq_len, seq_len))).bool(), toks_rep, -1)
        top_predicted_token_is_in_context: Bool[Tensor, "batch seqQ"] = (top_predicted_token_rep == toks_rep).any(dim=-1)
        batch_seqQ_indices = t.nonzero(top_predicted_token_is_non_fn_word & top_predicted_token_is_in_context)

        # If there are no destination tokens in the entire batch where the logit lens for this layer is a non-fn word in context, then skip
        if batch_seqQ_indices.numel() == 0:
            continue

        # Now I have all the (batch_idx, dest_idx) s.t. I actually want to take the result from that token
        batch_indices, seqQ_indices = batch_seqQ_indices.unbind(dim=-1)
        seqK_indices = (top_predicted_token[:, :, None] == toks[:, None, :]).int().argmax(dim=-1)[batch_indices, seqQ_indices]
        top_predicted_tokens = top_predicted_token[batch_indices, seqQ_indices]

        # if layer == 10:
        #     for b, sK, sQ in zip(batch_indices, seqK_indices, seqQ_indices):
        #         # if "Berk" in model.to_single_str_token(toks[b, sQ].item()) + model.to_single_str_token(toks[b, sK].item()):
        #         print(f"[{b:02}] Dest = {model.to_single_str_token(toks[b, sQ].item())!r}, Src = {model.to_single_str_token(toks[b, sK].item())!r}")

        # Get the actual things we're moving
        v_src: Float[Tensor, "batch_seqQ head d_head"] = v[batch_indices, seqK_indices]
        pattern_src_dest: Float[Tensor, "batch_seqQ head"] = pattern[batch_indices, :, seqQ_indices, seqK_indices]

        # Do the projections & attention scaling
        result_src: Float[Tensor, "batch_seqQ head d_model"] = einops.einsum(
            v_src, model.W_O[layer],
            "batch_seqQ head d_head, head d_head d_model -> batch_seqQ head d_model"
        )
        result_src_projections: Float[Tensor, "batch_seqQ head"] = einops.einsum(
            result_src, model.W_U.T[top_predicted_tokens],
            "batch_seqQ head d_model, batch_seqQ d_model -> batch_seqQ head"
        )
        scale = cache["scale"][batch_indices, seqQ_indices]
        result_dest_projections: Float[Tensor, "batch_seqQ head"] = (result_src_projections * pattern_src_dest) / scale

        # if layer == 10: print(batch_seqQ_indices)

        # Scaling by the number of nonzero elements (because I expect)
        results[layer] = einops.reduce(result_dest_projections, "batch_seqQ head -> head", "mean") * (toks.numel() / len(batch_indices))

    return results


# In[15]:


def get_in_vivo_copy_suppression_scores_3(model: HookedTransformer, DATA_STR: Int[Tensor, "batch seq"]):
    '''
    Tried projecting the attention (query-side) onto the unembeddings. I think this intervention just doesn't work though, so I can ditch it.
    '''

    toks = process_webtext_2(model, DATA_STR, SEQ_LEN)
    batch_size, seq_len = toks.shape

    _, cache = model.run_with_cache(
        toks,
        return_type = None,
        names_filter = lambda name: any(name.endswith(x) for x in ["v", "k", "resid_pre", "scale"])
    )

    FUNCTION_TOKS = t.concat([
        model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze().to(device),
        t.tensor([model.tokenizer.bos_token_id]).to(device)
    ])

    results = t.zeros((model.cfg.n_layers, model.cfg.n_heads))

    is_fn_word = (toks[:, :, None] == FUNCTION_TOKS[None, None, :]).any(dim=-1)

    for layer in range(model.cfg.n_layers):

        # Get everything we need from the cache
        resid_pre: Float[Tensor, "batch seqQ d_model"] = cache["resid_pre", layer]
        v: Float[Tensor, "batch seqK head d_head"] = cache["v", layer]
        k: Float[Tensor, "batch seqK head d_head"] = cache["k", layer]

        # Get projections so we can compute attention
        resid_pre_projected_onto_unembeddings = project(resid_pre, model.W_U.T[toks].unsqueeze(-1))
        # Compute attention, after doing this projection
        q = einops.einsum(resid_pre_projected_onto_unembeddings, model.W_Q[layer], "batch seqQ d_model, head d_model d_head -> batch seqQ head d_head") + model.b_Q[layer]
        attn_scores = einops.einsum(q, k, "batch seqQ head d_head, batch seqK head d_head -> batch seqQ seqK") / (model.cfg.d_head ** 0.5)
        attn_scores_masked = t.where(t.tril(t.ones((seq_len, seq_len))).bool(), attn_scores, -1e9)
        pattern = t.softmax(attn_scores_masked, dim=-1)

        # Set v to be zero wherever the token is a function word
        v_masked = t.where(is_fn_word[..., None, None], v, 0.0)
        # Get results, and project them onto the unembeddings for that source token
        result = einops.einsum(v_masked, model.W_O[layer], "batch seqK head d_head, head d_head d_model -> batch seqK head d_model")
        result_projected = project(result, einops.repeat(model.W_U.T[toks], "batch seqK d_head -> batch seqK head d_head 1", head=model.cfg.n_heads), only_keep="neg")

        # Get the norm
        result = einops.einsum(result_projected, pattern, "batch seqK head d_model, batch seqQ seqK -> batch seqQ head d_model")
        result_avg_norm: Float[Tensor, "head"] = result.pow(2).sum(dim=-1).sqrt().mean(dim=(0, 1))

        # Scaling by the number of nonzero elements (because I expect)
        results[layer] = result_avg_norm

    return results


# In[16]:


model.tokenizer.bos_token_id = model.tokenizer.eos_token_id

print("Getting CS scores...", end="\r")
t0 = time.time()
cs_ioi_scores = get_copy_suppression_scores_ioi(model, N=100)
print(f"Got CS scores in {time.time()-t0:.2f}s")

print("Getting AI scores...", end="\r")
t0 = time.time()
ai_rand_scores = get_anti_induction_scores(model, N=100)
print(f"Got AI scores in {time.time()-t0:.2f}s")

# TODO probably cut this one as we didn't plot it (?)???
if False:
    print("Getting in-vivo scores...", end="\r")
    t0 = time.time()
    in_vivo_scores = get_in_vivo_copy_suppression_scores_2(model, DATA_STR=DATA_STR)
    print(f"Got in-vivo scores in {time.time()-t0:.2f}s")
    all_results = t.stack([cs_ioi_scores, ai_rand_scores, in_vivo_scores])

else:
    all_results = t.stack([cs_ioi_scores, ai_rand_scores])

neg_results = all_results * (all_results < 0)
neg_results_01 = neg_results / einops.reduce(-neg_results, "stack layer head -> stack 1 1", "max")

imshow(
    neg_results_01,
    title = "Scores",
    facet_col = 0,
    facet_labels = ["Copy-suppression scores<br>(IOI)", "Anti-induction scores<br>(rand)", "Copy-suppression scores<br>(in-vivo)"][:2],
)


# In[22]:


# # all_results.shape

RESULTS_DIR = Path("/home/ubuntu/TransformerLens/transformer_lens/rs/callum2/st_page/media/anti_induction") if os.path.exists("/home/ubuntu") else Path("/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media/anti_induction")

with open(RESULTS_DIR / f"scores_{model.cfg.model_name}.pkl", "wb") as f:
    pickle.dump(all_results, f)


# In[18]:


def save_model_scores(model_name: str, N: int, plot: bool = False):

    t.cuda.empty_cache()

    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        device="cpu"
        # refactor_factored_attn_matrices=True,
    )
    model.set_use_attn_result(False)

    copy_suppression_scores_ioi = get_copy_suppression_scores_ioi(model, N)

    anti_induction_scores = get_anti_induction_scores(model, N)

    # in_vivo_copy_suppression_scores = get_in_vivo_copy_suppression_scores_2(model, DATA_STR)

    model_scores = t.stack([copy_suppression_scores_ioi, anti_induction_scores]) # , in_vivo_copy_suppression_scores])

    RESULTS_DIR = Path("/home/ubuntu/TransformerLens/transformer_lens/rs/callum2/st_page/media/anti_induction") if os.path.exists("/home/ubuntu") else Path("/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media/anti_induction")

    with open(RESULTS_DIR / f"scores_{model_name}.pkl", "wb") as f:
        pickle.dump(model_scores, f)

    if plot:
        neg_scores = model_scores * (model_scores < 0)
        neg_scores_01 = neg_scores / einops.reduce(neg_scores.abs(), "stack layer head -> stack 1 1", "max")
        imshow(
            neg_scores_01, # neg_model_scores,
            title=model_name,
            facet_col=0,
            text_auto=".1f",
            width=600 + (400 * model_scores.shape[0]),
            height=200 + (200 / 6) * model.cfg.n_layers,
            static=True,
            facet_labels=["IOI Copy Suppression DLA", "Anti-Induction DLA"] # , "CS-ablated norm"], # , "Funky weight scores"]
        )

def aggregate_saved_scores(
    overwrite: bool = False,
    delete: bool = False,
    show: bool = False,
):
    '''
    Aggregates all saved scores in the anti_induction folder, into a single dictionary `scores.pkl`.

    If `overwrite`, then it'll overwrite scores in the dictionary if they already exist.
    If `delete`, then it'll delete all the individual score files (leaving only the dict).
    '''
    RESULTS_DIR = Path("/home/ubuntu/TransformerLens/transformer_lens/rs/callum2/st_page/media/anti_induction") if os.path.exists("/home/ubuntu") else Path("/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media/anti_induction")
    results_dict_orig = pickle.load(open(RESULTS_DIR / "scores_dict.pkl", "rb"))
    results_dict_new = {
        file.stem.replace("scores_", ""): pickle.load(open(file, "rb"))
        for file in RESULTS_DIR.glob("scores_*.pkl")
        if "dict" not in file.stem
    }
    if overwrite:
        # The one on the right is the one which overrides the results
        results_dict = {**results_dict_orig, **results_dict_new}
    else:
        results_dict = {**results_dict_new, **results_dict_orig}

    if delete:
        for file in RESULTS_DIR.glob("scores_*.pkl"):
            file.unlink()

    if show:
        print("OLD")
        for k in sorted(results_dict_orig.keys()): print("  " + k)
        print("\nNEW")
        for k in sorted(results_dict_new.keys()): print("  " + k)

    pickle.dump(results_dict, open(RESULTS_DIR / "scores_dict.pkl", "wb"))


# In[23]:


aggregate_saved_scores(overwrite=True, delete=True, show=True)


# In[20]:


SMALL_MODEL_NAMES = [
    # "distillgpt2",
    "gpt2-small",
    # *[f"stanford-gpt2-small-{i}" for i in "abcde"],
    # *[f"pythia-{n}m" for n in [70, 160]],
    # *[f"pythia-{n}m-deduped" for n in [70, 160]],
    # *[f"solu-{n}l" for n in [4, 6, 8, 10]],
    # *[f"solu-{n}l-pile" for n in [4, 6, 8, 10]],
    # "gelu-4l",
    # "gpt-neo-125m",
    # "opt-125m",
]
MEDIUM_MODEL_NAMES = [
    "gpt-neo-125m",
    "gpt2-medium",
    *[f"stanford-gpt2-medium-{i}" for i in "abcde"],
    *[f"pythia-{n}m" for n in [410]],
    *[f"pythia-{n}m-deduped" for n in [410]],
    "solu-12l",
    "gpt2-large",
]
BIG_MODEL_NAMES = [
    *[f"pythia-{n}b" for n in [1.4, 2.8]],
    *[f"pythia-{n}b-deduped" for n in [1.4, 2.8]],
    "gpt2-xl",
    "gpt-neo-2.7B",
    "opt-1.3b",
    "opt-2.7b",
]
GIANT_MODEL_NAMES = [
    # *[f"pythia-{n}b" for n in [6.9]],
    # *[f"pythia-{n}b-deduped" for n in [6.9]],
    # "gpt-j-6B",
    # "opt-6.7b",
    "llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
]
BROBDINGNAGIAN_MODEL_NAMES = [
    *[f"pythia-{n}b" for n in [12]],
    *[f"pythia-{n}b-deduped" for n in [12]],
    "gpt-neox-20b",
    "opt-13b",
]


# In[33]:


# 1.3 mins (not inc. initial loading, metrics = CS/ioi + AI/rand + CS/norm)
for model_name in SMALL_MODEL_NAMES:
    t0 = time.time()
    save_model_scores(model_name, N=100, plot=True)
    print(f"Finished {model_name} in {time.time() - t0:.2f}s\n")


# In[ ]:


# 7.5 minutes (including initial model loading, metrics = CS/ioi + AI/rand)
# 5.9-3.9 minutes (not inc.  initial model loading, metrics = CS/ioi + AI/rand + CS/norm)
for model_name in MEDIUM_MODEL_NAMES:
    t0 = time.time()
    save_model_scores(model_name, N=100, plot=True)
    print(f"Finished {model_name} in {time.time() - t0:.2f}s\n")


# In[ ]:


# 10.9 minutes (including initial model loading, metrics = CS/ioi + AI/rand)
# 11.9-9.5 minutes (not including initial model loading, metrics = CS/ioi + AI/rand + CS/norm)
for model_name in BIG_MODEL_NAMES:
    t0 = time.time()
    save_model_scores(model_name, N=100, plot=False)
    print(f"Finished {model_name} in {time.time() - t0:.2f}s\n")


# In[116]:


# 20.6 minutes (including initial model loading)
for model_name in GIANT_MODEL_NAMES:
    t0 = time.time()
    save_model_scores(model_name, N=100, plot=False)
    print(f"Finished {model_name} in {time.time() - t0:.2f}s\n")


# In[114]:


for model_name in BROBDINGNAGIAN_MODEL_NAMES:
    t0 = time.time()
    save_model_scores(model_name, N=100, plot=False)
    print(f"Finished {model_name} in {time.time() - t0:.2f}s\n")


# In[ ]:


# new = "/home/ubuntu/Transformerlens/transformer_lens/rs/callum/anti_induction_vs_copy_suppression/model_results"
# old = "/home/ubuntu/Transformerlens/transformer_lens/rs/callum/streamlit/anti_induction_vs_copy_suppression/model_results"

# new = list(map(lambda x: x.name, Path(new).iterdir()))
# old = list(map(lambda x: x.name, Path(old).iterdir()))

# set(new) - set(old)
# set(old) - set(new)


# In[ ]:


import pandas as pd

def plot_all_results():
    results_copy_suppression_ioi = []
    results_anti_induction = []
    model_names = []
    head_names = []

    RESULTS_DIR = Path("/home/ubuntu/Transformerlens/transformer_lens/rs/callum/anti_induction_vs_copy_suppression/model_results")

    for file in RESULTS_DIR.iterdir():
        with open(file, "rb") as f:
            model_scores: Tensor = pickle.load(f)

            for layer in range(model_scores.size(1)):
                for head in range(model_scores.size(2)):
                    results_copy_suppression_ioi.append(model_scores[0, layer, head].item())
                    results_anti_induction.append(model_scores[1, layer, head].item())
                    model_names.append(file.stem.replace("scores_", ""))
                    head_names.append(f"{layer}.{head}")

    df = pd.DataFrame({
        "results_copy_suppression_ioi": results_copy_suppression_ioi,
        "results_anti_induction": results_anti_induction,
        "model_names": model_names,
        "head_names": head_names
    })

    fig = px.scatter(
        df,
        x="results_copy_suppression_ioi", y="results_anti_induction", color='model_names', hover_data=["model_names", "head_names"],
        width=1200,
        height=800,
        title="Anti-Induction Scores (repeated random tokens) vs Copy Suppression Scores (IOI)",
        labels={"results_copy_suppression_ioi": "Copy Suppression", "results_anti_induction": "Anti-Induction"}
    )
    fig.show()


plot_all_results()


# In[ ]:





# In[ ]:




