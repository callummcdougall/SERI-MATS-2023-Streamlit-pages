def generate_4_html_plots(
    model,
    data_toks,
    data_str_toks_parsed,
    negative_heads,
    save_files = False,
    model_results = None,
):
    '''
    Generates all the HTML plots for the Streamlit page. 

    This is called by me in `explore_prompts.ipynb`, to get data for open webtext.

    It's also called in the Streamlit page, to get data for the user's input.

    The output is in the form of nested dicts. Each key is a type of plot (e.g. the loss plot, or the logit attribution
    plot). Each value is itself a dict, containing all of these plots for different batch indices / types of ablation etc.
    '''
    HTML_PLOTS = {
        "LOSS": {},
        "LOGITS_ORIG": {},
        "LOGITS_ABLATED": {},
        "DLA": {},
        "ATTN": {},
        "UNEMBEDDINGS": {}
    }

    BATCH_SIZE = data_toks.shape[0]

    if model_results is None:
        MODEL_RESULTS = get_model_results(model, data_toks, negative_heads = negative_heads)
    else:
        MODEL_RESULTS = model_results

    # ! (1) Calculate the loss diffs from ablating

    loss_diffs = t.stack([
        t.stack(list(MODEL_RESULTS.loss.mean_direct.data.values())),
        t.stack(list(MODEL_RESULTS.loss.zero_direct.data.values())),
        t.stack(list(MODEL_RESULTS.loss.mean_patched.data.values())),
        t.stack(list(MODEL_RESULTS.loss.zero_patched.data.values())),
    ]) - MODEL_RESULTS.loss_orig

    # import time
    # t1 = 0

    for batch_idx in tqdm(range(BATCH_SIZE)):
        for head_idx, (layer, head) in enumerate(negative_heads):
            head_name = f"{layer}.{head}"

            # Calculate the loss diffs (and pad them with zero at the end, cause we don't know!)
            # shape is (ablation_type=4, batch, seq)
            loss_diffs_padded = t.concat([loss_diffs[:, head_idx], t.zeros((4, BATCH_SIZE, 1))], dim=-1)

            # For each different type of ablation, get the loss diffs
            # t0 = time.time()
            for loss_diff, ablation_type in zip(loss_diffs_padded, ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]):
                html_25, html_max = generate_html_for_loss_plot(
                    data_str_toks_parsed[batch_idx],
                    loss_diff = loss_diff[batch_idx],
                )
                HTML_PLOTS["LOSS"][(batch_idx, head_name, ablation_type, True)] = str(html_max)
                HTML_PLOTS["LOSS"][(batch_idx, head_name, ablation_type, False)] = str(html_25)
            # t1 += (time.time() - t0)

    # print(t1)

    # ! (2, 3, 4) Calculate the logits & direct logit attributions

    token_log_probs_dict = {
        "orig": MODEL_RESULTS.logits_orig.log_softmax(-1),
        **{
            f"{ablation_type}, {layer}.{head}": getattr(MODEL_RESULTS.logits, ablation_type.replace(", ", "_"))[layer, head].log_softmax(-1)
            for layer, head in negative_heads
            for ablation_type in ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]
        }
    }
    token_log_probs_top10_dict = {
        k: v.topk(10, dim=-1)
        for (k, v) in token_log_probs_dict.items()
    }
    direct_effect_log_probs_dict = {
        (layer, head): MODEL_RESULTS.direct_effect[layer, head].log_softmax(-1)
        for layer, head in negative_heads
    }

    for batch_idx in tqdm(range(BATCH_SIZE)):

        html_orig = generate_html_for_logit_plot(
            data_toks,
            token_log_probs_dict["orig"],
            token_log_probs_top10_dict["orig"],
            token_log_probs_dict["orig"],
            batch_idx,
            model,
        )
        HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)] = str(html_orig)

        for (layer, head) in negative_heads:
            head_name = f"{layer}.{head}"

            # Save new log probs (post-ablation)
            for ablation_type in ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]:

                html_ablated = generate_html_for_logit_plot(
                    data_toks,
                    token_log_probs_dict[f"{ablation_type}, {layer}.{head}"],
                    token_log_probs_top10_dict[f"{ablation_type}, {layer}.{head}"],
                    token_log_probs_dict["orig"],
                    batch_idx,
                    model,
                )
                HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, ablation_type)] = str(html_ablated)
            
            # # Save direct logit effect
            dla_neg, dla_pos = generate_html_for_DLA_plot(
                data_toks[batch_idx],
                direct_effect_log_probs_dict[(layer, head)][batch_idx],
                model
            )
            HTML_PLOTS["DLA"][(batch_idx, head_name, "neg")] = str(dla_neg)
            HTML_PLOTS["DLA"][(batch_idx, head_name, "pos")] = str(dla_pos)
    

    # ! (5) Calculate the attention probs

    for batch_idx in tqdm(range(BATCH_SIZE)):

        for layer, head in negative_heads:
            head_name = f"{layer}.{head}"

            # Calculate attention, and info-weighted attention
            attn = MODEL_RESULTS.pattern[layer, head][batch_idx]
            weighted_attn = einops.einsum(
                MODEL_RESULTS.pattern[layer, head][batch_idx],
                MODEL_RESULTS.out_norm[layer, head][batch_idx] / MODEL_RESULTS.out_norm[layer, head][batch_idx].max(),
                "seqQ seqK, seqK -> seqQ seqK"
            )

            for vis_name, vis_type in {"Large": cv.attention.attention_heads, "Small": cv.attention.attention_patterns}.items():
                html_standard, html_weighted = [
                    vis_type(
                        attention = x.unsqueeze(0), # (heads=2, seqQ, seqK)
                        tokens = data_str_toks_parsed[batch_idx], # list of length seqQ
                        attention_head_names = [head_name]
                    )
                    for x in [attn, weighted_attn]
                ]
                html_standard, html_weighted = list(map(attn_filter, [html_standard, html_weighted]))
                HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, "standard")] = str(html_standard)
                HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, "info-weighted")] = str(html_weighted)

            
    # ! (6) Calculate the component of the unembeddings in pre-head residual stream, 

    str_toks = [model.to_str_tokens(seq) for seq in data_toks]

    for layer, head in negative_heads:
        head_name = f"{layer}.{head}"
        # Get the unembedding components in resid_pre just before this head
        W_U_comp_avg = MODEL_RESULTS.unembedding_components[layer]["avg"]
        W_U_comp_top10 = MODEL_RESULTS.unembedding_components[layer]["top10"]
        # Generate the HTML for these components (separate for including self and excluding self)
        html_dict = generate_html_for_unembedding_components_plot(str_toks, W_U_comp_avg[0], W_U_comp_top10[0])
        html_rm_self_dict = generate_html_for_unembedding_components_plot(str_toks, W_U_comp_avg[1], W_U_comp_top10[1])
        # Add these to dictionary, all at once
        HTML_PLOTS["UNEMBEDDINGS"] = {
            **HTML_PLOTS["UNEMBEDDINGS"],
            # **{(batch_idx, head_name, True): html for batch_idx, html in html_rm_self_dict.items()},
            **{(batch_idx, head_name): html for batch_idx, html in html_dict.items()},
        }

    # Optionally, save the files (we do this if we're generating it from OWT, for the Streamlit page)
    if save_files:
        with gzip.open(ST_HTML_PATH / "GZIP_HTML_PLOTS.pkl", "wb") as f:
            pickle.dump(HTML_PLOTS, f)

    return HTML_PLOTS

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cpu"
    # refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(False)
model.set_use_attn_result(True)

clear_output()

from generate_html import (
    generate_html_for_logit_plot,
    generate_html_for_DLA_plot,
    generate_html_for_unembedding_components_plot,
    generate_html_for_loss_plot,
    attn_filter,
)
from model_results import ModelResults, get_model_results

MODEL_RESULTS = get_model_results(model, DATA_TOKS, negative_heads = NEGATIVE_HEADS)

HTML_PLOTS = generate_4_html_plots(
    model_results = MODEL_RESULTS,
    model = model,
    data_toks = DATA_TOKS,
    data_str_toks_parsed = DATA_STR_TOKS_PARSED,
    negative_heads = [(10, 7), (11, 10)],
    save_files = True,
)
