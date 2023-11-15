# In[1]:

import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
from transformer_lens.cautils.notebook import *
t.set_grad_enabled(False)
import argparse
import transformers
from transformer_lens import HookedTransformer # Some TODOs; script this so we can sweep all the darn models. Also, check that code isn't bugged; ensure things work normally for GPT-2 Small
import torch
import wandb
from transformer_lens.rs.callum2.utils import (
    create_title_and_subtitles,
    parse_str,
    parse_str_tok_for_printing,
    parse_str_toks_for_printing,
    topk_of_Nd_tensor,
    ST_HTML_PATH,
    project,
)
from transformer_lens import FactoredMatrix
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
    "distilgpt2": "42M",
    "distillgpt2": "42M",
    "opt-125m": "85M",
    "opt-1.3b": "1.2B",
    "opt-2.7b": "2.5B",
    "opt-6.7b": "6.4B",
    "opt-13b": "13B",
    "gpt-neo-125m": "85M",
    "gpt-neo-1.3b": "1.2B",
    "gpt-neo-2.7b": "2.5B",
    "gpt-neo-1.3B": "1.2B",
    "gpt-neo-2.7B": "2.5B",
    "gpt-j-6B": "5.6B",
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
    # "mistralai/Mistral-7B-v0.1": "7B", # Broken, sad
    "Llama-2-7b-hf": "7B",
    "Llama-2-13b-hf": "13B",
    "llama-7b-hf": "7B",
    "llama-13b-hf": "13B",
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

if ipython is None:
    # Parse all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pythia-410m")
    args = parser.parse_args()
    model_name = args.model
    wandb_notes = ""

    try:
        with open(__file__, "r") as f:
            wandb_notes += f.read()
    except Exception as e:
        print("Couldn't read file", __file__, "due to", str(e), "so not adding notes")
    run_name = model_name
    wandb.init(
        project="copy_suppression_induction_ioi",
        job_type="train",
        name=run_name,
        mode="online",
        notes=wandb_notes,
    )   

else:
    model_name = "pythia-410m"

def plot_all_results(
    pospos=False,
    showtext=False,
    fraction=False,
    categories="all",
    # x = "results_cs_norm"
    x = "results_cs_ioi",
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
        title="Anti-Induction Scores (repeated random tokens) vs Copy-Suppression Scores " + ("(IOI)" if x == "results_cs_ioi" else "(OWT)"),
        labels={
            x: "Copy-Suppression Score (IOI)" if x == "results_cs_ioi" else "Copy-Suppression Score (OWT)",
            "results_ai_rand": "Anti-Induction Score",
            "model_class": "Model Class",
        },
        height=650,
        width=1000,
        color_continuous_scale=px.colors.sequential.Rainbow if fraction else None,
        template="simple_white",
    )
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
    return fig

# In[5]:

fig = plot_all_results(
    pospos=False,
    showtext=False,
    fraction=False,
    categories="all",
    x = "results_cs_norm",
)
fig.update_layout(height=550, width=800)
MEDIA_PATH = Path(r"/home/ubuntu/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media") if os.path.exists("/home/ubuntu") else Path(r"/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media")
from plotly.utils import PlotlyJSONEncoder
import json
with open(MEDIA_PATH / "ai_vs_cs.json", 'w') as f:
    f.write(json.dumps(fig, cls=PlotlyJSONEncoder))
# fig.write_image(media_path / "ai_vs_cs.pdf")
    
if ipython is None:
    # Log this image to wandb
    wandb.log({"ai_vs_cs_owt": wandb.Image(fig)})

else:
    fig.show()


# In[6]:

fig = plot_all_results(
    pospos=False,
    showtext=False,
    fraction=False,
    categories="all",
)

fig.write_image(media_path / "ai_vs_cs.pdf")

if ipython is None:
    # Log this image to wandb
    wandb.log({"ai_vs_cs_ioi": wandb.Image(fig)})

else:
    fig.show()

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
    # model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "meta-llama/Llama-2-13b-hf"
    # model_name = "distillgpt2"
    # model_name = "pythia-410m"
    # model_name = "gpt2-small"
    # from_pretrained version 4 minutes

    long_path = "/workspace/cache/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658/"
    if "llama" in model_name.lower() or "mistral" in model_name.lower():
        if "model" in locals():
            del model
            gc.collect()
            torch.cuda.empty_cache()
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(long_path) if "mistral" in model_name.lower() and os.path.exists(long_path) else transformers.AutoModelForCausalLM.from_pretrained(model_name)
        gc.collect()
        torch.cuda.empty_cache()
        print("...")
        model = hf_model.to(torch.bfloat16)
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        print("Got tokenizer...")
        gc.collect()
        torch.cuda.empty_cache()
        model = HookedTransformer.from_pretrained_no_processing(model_name, hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=hf_tokenizer, dtype=torch.bfloat16)
        print("Got model in TL...")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model = HookedTransformer.from_pretrained(model_name, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)

    torch.set_grad_enabled(False)
    gc.collect()
    torch.cuda.empty_cache()
    model = model.to(torch.bfloat16)
    gc.collect()
    torch.cuda.empty_cache()
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

# In[12]:

BATCH_SIZE = 1000 # 91 for scatter, 51 for viz
SEQ_LEN = 1000 # 100 for scatter, 61 for viz (no more, cause attn)

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
minibatchsize = 5
assert BATCH_SIZE % minibatchsize == 0, f"Batch size {BATCH_SIZE} must be divisible by minibatch size {minibatchsize}"

# In[13]:

# if True:
def get_in_vivo_copy_suppression_scores_2(model: HookedTransformer, DATA_STR: Int[Tensor, "batch seq"]):
    '''
    Same as the other one, except rather than picking the top source token, it picks the top token over all 50k words, and sets the result to zero
    if that top token isn't in context. This is a lot more strict, and hopefully a lot more sparse.
    '''

    results = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device, dtype=t.float)

    for minibatch_strs in tqdm([
        DATA_STR[i:i+minibatchsize] for i in range(0, BATCH_SIZE, minibatchsize)
    ]):
        toks = process_webtext_2(model, minibatch_strs, SEQ_LEN)
        _, seq_len = toks.shape

        logits, cache = model.run_with_cache(
            toks,
            names_filter = lambda name: any(name.endswith(x) for x in ["pattern", "v", "resid_pre", "scale"])
        )

        # Sanity check loss is low
        logprobs = t.nn.functional.log_softmax(logits, dim=-1)
        loss = -logprobs[torch.arange(minibatchsize)[:, None], torch.arange(seq_len-1)[None, :], toks[:, 1:]].mean()
        print("FYI, loss", round(loss.item(), 3))

        potential_function_toks = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze().to(device)
        FUNCTION_TOKS = t.concat([t.tensor([model.tokenizer.bos_token_id]).to(device)] + ([] if len(potential_function_toks.shape)>1 else [potential_function_toks]) + ([torch.tensor([1]).to(device)] if "llama" in model.cfg.model_name.lower() or "mistral" in model.cfg.model_name.lower() else []))

        for layer in range(model.cfg.n_layers):
            # Get everything we need from the cache
            resid_post_scaled: Float[Tensor, "batch seqQ d_model"] = cache["resid_pre", layer] / cache["scale"]
            v: Float[Tensor, "batch seqK head d_head"] = cache["v", layer]
            pattern: Float[Tensor, "batch head seqQ seqK"] = cache["pattern", layer]

            # Get logit lens for all tokens, at each dest token
            # Get the (batch, seqQ) indices of everywhere that the top token isn't a function word & is in context
            logit_lens: Float[Tensor, "batch seqQ d_vocab"] = resid_post_scaled.to(model.cfg.dtype) @ model.W_U
            top_predicted_token: Int[Tensor, "batch seqQ"] = logit_lens.argmax(-1)
            top_predicted_token_is_non_fn_word: Bool[Tensor, "batch seqQ"] = (top_predicted_token[:, :, None] != FUNCTION_TOKS[None, :]).all(dim=-1)
            top_predicted_token_rep = einops.repeat(top_predicted_token, "batch seqQ -> batch seqQ seqK", seqK=seq_len)
            toks_rep = einops.repeat(toks, "batch seqK -> batch seqQ seqK", seqQ=seq_len)
            toks_rep = t.where(t.tril(t.ones((seq_len, seq_len))).bool().to(toks_rep.device), toks_rep, -1)
            top_predicted_token_is_in_context: Bool[Tensor, "batch seqQ"] = (top_predicted_token_rep == toks_rep).any(dim=-1)
            batch_seqQ_indices = t.nonzero(top_predicted_token_is_non_fn_word & top_predicted_token_is_in_context)

            # If there are no destination tokens in the entire batch where the logit lens for this layer is a non-fn word in context, then skip
            if batch_seqQ_indices.numel() == 0:
                continue

            # Now I have all the (batch_idx, dest_idx) s.t. I actually want to take the result from that token
            batch_indices, seqQ_indices = batch_seqQ_indices.unbind(dim=-1)
            seqK_indices = (top_predicted_token[:, :, None] == toks[:, None, :]).int().argmax(dim=-1)[batch_indices, seqQ_indices]
            top_predicted_tokens = top_predicted_token[batch_indices, seqQ_indices]

            if seqK_indices.min().item()==0:
                warnings.warn("Uhok we found zeros this isnt good")

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
            results[layer] += einops.reduce(result_dest_projections, "batch_seqQ head -> head", "mean") * (toks.numel() / len(batch_indices))

    returning = results / (BATCH_SIZE // minibatchsize)
    # print(returning)
    return returning

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
if True:
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

RESULTS_DIR = Path("/home/ubuntu/TransformerLens/transformer_lens/rs/callum2/st_page/media/anti_induction") if os.path.exists("/home/ubuntu") else Path("/workspace/SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page/media/anti_induction")

with open(RESULTS_DIR / f"scores_{model.cfg.model_name}.pkl", "wb") as f:
    pickle.dump(all_results, f)

# Save that file to wandb as an artifact
if ipython is None:
    artifact = wandb.Artifact(
        name=f"anti_induction_scores_{model.cfg.model_name}",
        type="anti_induction_scores",
        description="Scores for anti-induction",
    )
    artifact.add_file(RESULTS_DIR / f"scores_{model.cfg.model_name}.pkl")
    wandb.log_artifact(artifact)

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

#%%


