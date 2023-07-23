# This contains functions to generate HTML for each of the different Streamlit visualisations.

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

import gzip
from typing import Tuple, List, Any, Optional, Dict
import torch as t
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from IPython.display import display, HTML
from plotly.colors import sample_colorscale
import einops
from tqdm import tqdm
import circuitsvis as cv
import pickle
import pandas as pd
import numpy as np

from explore_prompts_utils import (
    ST_HTML_PATH,
    NEGATIVE_HEADS
)
from model_results_3 import (
    ModelResults,
    get_model_results
)

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@350&display=swap" rel="stylesheet">

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>

<style>
body {
    font-family: 'Source Sans 3', sans-serif;
}

table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid black;
    padding: 6px;
    text-align: left;
    line-height: 0.8em;
}
.empty-row td {
    border: none;
}
mark {
    font-size: 1rem;
    line-height: 1.8rem;
    padding: 1px;
    margin-right: 1px;
}
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}
.tooltip .tooltiptext {
    font-size: 0.9rem;
    min-width: 275px;
    visibility: hidden;
    background-color: #eee;
    color: #000;
    text-align: center;
    padding: 5px;
    position: absolute;
    z-index: 1;
    top: 125%;
    left: 50%;
    margin-left: -10px;
    opacity: 0;
    transition: opacity 0.0s;
}
.tooltip.hovered .tooltiptext,
.tooltip.clicked .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>

<script>
$(document).ready(function(){
  $('.tooltip').hover(function(){
    if (!$(this).hasClass('clicked')) {
      $(this).addClass('hovered');
    }
    var tooltipWidth = $(this).children('.tooltiptext').outerWidth();
    var viewportWidth = $(window).width();
    var tooltipRight = $(this).offset().left + tooltipWidth;
    if (tooltipRight > viewportWidth) {
      $(this).children('.tooltiptext').css('left', 'auto').css('right', '0');
    }
  }, function() {
    if (!$(this).hasClass('clicked')) {
      $(this).removeClass('hovered');
    }
    $(this).children('.tooltiptext').css('left', '50%').css('right', 'auto');
  });

  $('.tooltip').click(function(e){
    e.stopPropagation();
    if ($(this).hasClass('clicked')) {
      $(this).removeClass('clicked');
    } else {
      $('.tooltip').removeClass('clicked');
      $(this).addClass('clicked');
    }
  });

  $(document).click(function() {
    $('.tooltip').removeClass('clicked');
  });
});
</script>
"""

import re

def attn_filter(html):
    html = str(html)
    def round_match(match):
        return "{:.4f}".format(float(match.group()))
    return re.sub(r'\b0\.\d+\b', round_match, html)




def _get_color(importances):
    """
    Returns a color based on the importance of a word.

    Also returns color for the text (has to be white if sufficiently extreme).    
    """
    bg_colors = sample_colorscale("RdBu", importances, low=0.0, high=1.0, colortype='rgb')
    text_colors = list(map(lambda i: "white" if abs(i - 0.5) > 0.3 else "black", importances))
    return bg_colors, text_colors



def format_word_importances(
    words: List[str],
    importances: List[float],
    hover_text_list: List[str],
) -> str:
    """Adds a background color to each word based on its importance (float from -1 to 1)

    Args:
        words (list): List of words
        importances (list): List of importances (scores from -1 to 1)

    Returns:
        html: HTML string with formatted word
    """
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) == len(importances), "Words and importances but be of same length"

    tags = ["<td>"]
    bg_colors, text_colors = _get_color(importances)
    for word, bg_color, text_color, hover_text in zip(words, bg_colors, text_colors, hover_text_list):
        word = word.replace(" ", "&nbsp;")
        if "\n" in word:
            word = "\\n"
        unwrapped_tag = f'<mark style="background-color:{bg_color}"><font color="{text_color}">{word}</font></mark>'
        unwrapped_tag = f'<span class="tooltip">{unwrapped_tag}<span class="tooltiptext">{hover_text}</span></span>'
        tags.append(unwrapped_tag)
    tags.append("</td>")
    html = "&nbsp;".join(tags)
    return html



def generate_html_for_loss_plot(
    str_toks: List[str],
    loss_diff: Float[Tensor, "seq"],
):
    '''
    Returns HTML for the loss plot (i.e. you hover over a word and it tells you how much
    the loss has increased or decreased by when you ablate).
    '''
    # Chosen 2.5 as an appropriate "extreme loss" baseline
    max_colors = [2.5, loss_diff.abs().max().item() + 1e-6]

    importances = [t.clip(0.5 + loss_diff / (2 * max_color), 0.0, 1.0).tolist() for max_color in max_colors]

    hover_text_list = [f"({i}) <b>'{s}'</b><br>{d:.4f}" for i, (s, d) in enumerate(zip(str_toks[:-1], loss_diff))] + [""]

    html_25 = format_word_importances(str_toks, importances[0], hover_text_list)
    html_max = format_word_importances(str_toks, importances[1], hover_text_list)
    return html_25, html_max



def generate_html_for_unembedding_components_plot(
    str_toks: List[List[str]], # shape (batch, seq) really
    unembedding_components_avg: Float[Tensor, "batch seq"],
    unembedding_components_top10, # topk return type, shape (batch, seq, 10)
):
    results = {}

    for batch_idx in range(len(str_toks)):

        importances = []
        hover_text_list = []

        max_value = unembedding_components_top10.values[batch_idx, :, :].max().item()

        for seq_pos in range(unembedding_components_avg.size(1)):
            avg = unembedding_components_avg[batch_idx, seq_pos].item()
            top_values = unembedding_components_top10.values[batch_idx, seq_pos, :].tolist()
            top_indices = unembedding_components_top10.indices[batch_idx, seq_pos, :].tolist()
            top_words = [str_toks[batch_idx][idx] for idx in top_indices]

            max_value_minus_avg = max_value - avg

            table_body = ""
            for idx, (word, value) in enumerate(zip(top_words, top_values)):
                if value < -1e8: break
                table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{value:.2f}</td></tr>"
                if idx == 0: importances.append(max(0, value-avg) / max_value_minus_avg)
            if table_body == "":
                importances.append(0.0)

            avg_row = f"<tr><td>AVG</td><td></td><td>{avg:.2f}</td></tr>".replace('<td>', '<td style="background-color:#c2d9ff; color:black">')
            hover_text_list.append("".join([
                "<table>",
                "<thead><tr><th>Rank</th><th>Word</th><th>W<sub>U</sub>-component</th></tr></thead>",
                f"<tbody>{table_body}{avg_row}</tbody></table>"
            ]))

        importances = [0.5 + imp / (2 * max(importances)) for imp in importances]

        html = format_word_importances(str_toks[batch_idx], importances, hover_text_list)

        results[batch_idx] = html

    return results



def generate_html_for_logit_plot(
    toks: Int[Tensor, "batch seq"],
    logprobs: Float[Tensor, "batch seq d_vocab"],
    logprobs_top10: Any,
    non_ablated_logprobs: Float[Tensor, "batch seq d_vocab"],
    batch_idx: int,
    model: HookedTransformer,
    logprobs_top5_in_ctx: Any = None,
):
    '''
    This gets the top predicted tokens for the model (and the prediction for the actual next token).

    We use non_ablated_logits as a baseline. The baseline tokens will be colored by the logprob of the
    token which actually came next, the non-baseline tokens will be colored by the difference in this prob
    from the baseline to them.

    If logprobs_top5_in_ctx is given, then we also add 5 rows to the end of the hovertext, which correspond
    to the top 5 predictions for the next token WHICH ALSO APPEAR IN CONTEXT. This is useful for the logit
    lens visualisations, when we want to verify prediction-attention is happening
    '''
    toks = toks[batch_idx]
    logprobs = logprobs[batch_idx]
    non_ablated_logprobs = non_ablated_logprobs[batch_idx]
    seq_len = logprobs.size(0)

    max_color = 4 # This is the fixed point for the color scale

    # Get the thing I'll use to color the tokens (leaving white if there's basically no difference)
    str_toks = model.to_str_tokens(toks)
    logprobs_on_correct_token = logprobs[range(len(str_toks)-1), toks[1:]]
    logprobs_on_correct_token_baseline = non_ablated_logprobs[range(len(str_toks)-1), toks[1:]]
    probs_on_correct_token = logprobs_on_correct_token.exp()
    colors = logprobs_on_correct_token - logprobs_on_correct_token_baseline
    if colors.abs().max() < 1e-4:
        # In this case, we must have the original logprobs, so we set colors based on just these
        orig = True
        # 0.5 is the baseline for "very low probability" (logprob=-4), we get 1.0 close to logprob=0
        colors = 1 + t.maximum(
            logprobs_on_correct_token / (2 * max_color),
            t.full_like(logprobs_on_correct_token, -0.5)
        )
    else:
        # In this case, these must be ablated logprobs, so we set colors based on the difference
        orig = False
        diff = logprobs_on_correct_token - logprobs_on_correct_token_baseline
        # 0.5 is the baseline for "no change in logprob", we get 1.0 or 0.0 for a change of ±4
        colors = t.clip(0.5 + diff / (2 * max_color), 0.0, 1.0)
    colors = colors.tolist() + [0.5]

    # Now, get the top10 predictions for each token
    str_toks_top10_all = [model.to_str_tokens(logprobs_top10.indices[batch_idx, i]) for i in range(seq_len)]
    logprobs_top10_all = logprobs_top10.values[batch_idx]
    if logprobs_top5_in_ctx is not None:
        max_color = 30
        seqpos_top5_ctx = logprobs_top5_in_ctx.indices[batch_idx] # shape (seqQ, 5)
        str_toks_top5_ctx_all = [model.to_str_tokens(toks[seqpos_top5_ctx[i]]) for i in range(seq_len)]
        logprobs_top5_ctx_all = logprobs_top5_in_ctx.values[batch_idx] # shape (seqQ, 5)

    # Now, get the hovertext for my formatting function

    hover_text_list = []

    all_correct_ranks = (logprobs_on_correct_token.unsqueeze(1) < logprobs[range(len(str_toks)-1)]).sum(dim=1)
    if logprobs_top5_in_ctx is not None:
        logprobs_top5_ctx_all_rep = einops.repeat(logprobs_top5_ctx_all, 'seqQ K -> seqQ 1 K')
        logprobs_rep = einops.repeat(logprobs, 'seqQ d_vocab -> seqQ d_vocab 1')
        all_top5_ctx_ranks: Int[Tensor, "seqQ K"] = (logprobs_top5_ctx_all_rep < logprobs_rep).sum(dim=1)

    for seq_pos in range(len(str_toks) - 1):

        current_word = str_toks[seq_pos]
        next_word = str_toks[seq_pos + 1]

        correct_rank = all_correct_ranks[seq_pos]
        str_toks_top10 = str_toks_top10_all[seq_pos]
        logprobs_top10 = logprobs_top10_all[seq_pos]
        probs_top10 = logprobs_top10.exp()
        if logprobs_top5_in_ctx is not None:
            str_toks_top5_ctx = str_toks_top5_ctx_all[seq_pos]
            logprobs_top5_ctx = logprobs_top5_ctx_all[seq_pos]
            probs_top5_ctx = logprobs_top5_ctx.exp()
            top5_ctx_ranks = all_top5_ctx_ranks[seq_pos]


        table_body = ""
        for idx, (word, logprob, prob) in enumerate(zip(str_toks_top10, logprobs_top10, probs_top10)):
            table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{logprob:.2f}</td><td>{prob:.2%}</td></tr>"

        lp_orig = logprobs_on_correct_token_baseline[seq_pos]
        lp = logprobs_on_correct_token[seq_pos]
        p = probs_on_correct_token[seq_pos]
        empty_row = '<tr class="empty-row"><td></td><td></td><td></td><td></td></tr>'
        new_hover_text = "".join([
            "<table>",
            "<thead><tr><th>Rank</th><th>Word</th><th>Logprob</th><th>Prob</th></tr></thead>",
            f"<tbody><tr><td>#{correct_rank}</td><td>{next_word!r}</td><td>{lp:.2f}</td><td>{p:.2%}</td></tr>"
        ]).replace('<td>', '<td style="background-color:#c2d9ff; color:black">') + f"{empty_row}{table_body}</tbody></table>"
        if not(orig):
            new_hover_text = f"Δ logprob on correct token = {lp-lp_orig:.2f}<br><br>" + new_hover_text

        # Finally, get the top5 in context thing, if we are given it
        if logprobs_top5_in_ctx is not None:
            logprobs_mean = logprobs[seq_pos].mean()
            table_body = ""
            for (rank, word, logprob, prob) in zip(top5_ctx_ranks, str_toks_top5_ctx, logprobs_top5_ctx, probs_top5_ctx):
                if logprob > -1e4:
                    # * Not coloring in this way because it's visually unclear and confusing (e.g. function words, what baseline to use).
                    # colors[seq_pos] = min(1, 0.5 + (logprob - logprobs_mean).item() / (2 * max_color))
                    table_body += f"<tr><td>#{rank}</td><td>{word!r}</td><td>{logprob:.2f}</td><td>{prob:.2%}</td></tr>"
            new_hover_text += f"<br><b>Top 5 predictions from context (avg = {logprobs_mean:.2f}):</b><br><br>" + "".join([
                "<table>",
                "<thead><tr><th>Rank</th><th>Word</th><th>Logprob</th><th>Prob</th></tr></thead>",
                f"<tbody>{table_body}</tbody></table>"
            ])

        hover_text_list.append(f"<span background-color:'#ddd'>{current_word!r}</span> ➔ <span background-color:'#ddd'>{next_word!r}</span><br><br>{new_hover_text}")

    hover_text_list = hover_text_list + ["(next token unknown)"]
    html = format_word_importances(str_toks, colors, hover_text_list)

    return html



def generate_html_for_DLA_plot(
    toks: Int[Tensor, "seq_len"],
    dla_logits: Float[Tensor, "seq_len-1 d_vocab"],
    model: HookedTransformer,
):
    dla_logits_topk = dla_logits.topk(dim=-1, k=10)
    dla_logits_botk = dla_logits.topk(dim=-1, k=10, largest=False)

    # * From eyeballing data, it looks like logit suppression of 2.5 is a sufficiently extreme value
    # * for it to be reasonable setting this as the default upper color.
    importances_pos_raw = dla_logits.max(dim=-1).values.tolist()
    importances_neg_raw = (-dla_logits).max(dim=-1).values.tolist()
    denom = 2 * max([2.5] + importances_pos_raw + importances_neg_raw)

    importances_pos = [0.5 + imp / denom for imp in importances_pos_raw]
    importances_neg = [0.5 - imp / denom for imp in importances_neg_raw]

    words = model.to_str_tokens(toks) + ["(can't see next token)"]

    hover_text_list_neg = []
    hover_text_list_pos = []

    for i in range(len(toks)):
        current_word = words[i].replace(" ", "&nbsp;")
        next_word = words[i+1].replace(" ", "&nbsp;")

        top10_values = dla_logits_topk.values[i].tolist()
        top10_indices = dla_logits_topk.indices[i].tolist()
        top10_str_toks = list(map(model.to_single_str_token, top10_indices))
        bot10_values = dla_logits_botk.values[i].tolist()
        bot10_indices = dla_logits_botk.indices[i].tolist()
        bot10_str_toks = list(map(model.to_single_str_token, bot10_indices))

        neg_table_body = ""
        pos_table_body = ""
        for idx, (word, value) in enumerate(zip(bot10_str_toks, bot10_values)):
            neg_table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{value:.2f}</td></tr>"
        for idx, (word, value) in enumerate(zip(top10_str_toks, top10_values)):
            pos_table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{value:.2f}</td></tr>"

        hover_text_list_neg.append("".join([
            f"<span background-color:'#ddd'>{current_word!r}</span> ➔ <span background-color:'#ddd'>{next_word!r}</span><br><br>",
            "<table>",
            "<thead><tr><th>Rank</th><th>Word</th><th>Logit</th></tr></thead>",
            f"<tbody>{neg_table_body}</tbody></table>",
        ]))
        hover_text_list_pos.append("".join([
            f"<span background-color:'#ddd'>{current_word!r}</span> ➔ <span background-color:'#ddd'>{next_word!r}</span><br><br>",
            "<table>",
            "<thead><tr><th>Rank</th><th>Word</th><th>Logit</th></tr></thead>",
            f"<tbody>{pos_table_body}</tbody></table>"
        ]))

    html_neg = format_word_importances(words[:-1], importances_neg, hover_text_list_neg)
    html_pos = format_word_importances(words[:-1], importances_pos, hover_text_list_pos)

    return html_neg, html_pos




def generate_4_html_plots(
    model: HookedTransformer,
    data_toks: Float[Int, "batch seq_len"],
    data_str_toks_parsed: List[List[str]],
    negative_heads: List[Tuple[int, int]] = NEGATIVE_HEADS,
    save_files: bool = False,
    model_results: Optional[ModelResults] = None,
    progress_bar: bool = False,
    restrict_computation: List[str] = ["LOSS", "LOGITS", "ATTN", "UNEMBEDDINGS"],
    cspa: bool = True,
) -> Dict[str, Dict[Tuple, str]]:
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
        model_results = get_model_results(model, data_toks, negative_heads=negative_heads, cspa=cspa)

    # ! (1) Calculate the loss diffs from ablating

    if "LOSS" in restrict_computation:

        for batch_idx in (tqdm(range(BATCH_SIZE), "LOSS") if progress_bar else range(BATCH_SIZE)):
            for (layer, head) in negative_heads:
                head_name = f"{layer}.{head}"

                # For each different type of ablation, get the loss diffs
                for effect in ["direct", "indirect", "both"]:
                    for ln_mode in ["frozen", "unfrozen"]:
                        for ablation_mode in ["zero", "mean"]:
                            # Get the loss per token, and pad with zeros at the end (cause we don't know the last value!)
                            loss_diff = model_results.loss_diffs[(effect, ln_mode, ablation_mode)][layer, head][batch_idx]
                            loss_diff_padded = t.cat([loss_diff, t.zeros((1,))], dim=-1)
                            html_25, html_max = generate_html_for_loss_plot(
                                data_str_toks_parsed[batch_idx],
                                loss_diff = loss_diff_padded,
                            )
                            full_ablation_mode = "+".join([effect, ln_mode, ablation_mode])
                            HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_mode, True)] = str(html_max)
                            HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_mode, False)] = str(html_25)


    # ! (2, 3, 4) Calculate the logits & direct logit attributions
    # For logprobs, we batch this - first calculating all the logprobs and the top10s, then passing them all into the `generate_html_for_logit_plot` 
    # function along with a `batch_idx` argument.
    # For DLA, we don't batch it, i.e. we just pass in the vector of DLA logits.

    if "LOGITS" in restrict_computation:

        LOGPROBS_DICT = {
            "orig": model_results.logits_orig.log_softmax(-1),
            **{
                ('+'.join([effect, ln_mode, ablation_mode]), f"{layer}.{head}"): model_results.logits[(effect, ln_mode, ablation_mode)][layer, head].log_softmax(-1)
                for layer, head in negative_heads
                for effect in ["direct", "indirect", "both"]
                for ln_mode in ["frozen", "unfrozen"]
                for ablation_mode in ["zero", "mean"]
            }
        }
        LOGPROBS_TOP10_DICT = {
            k: v.topk(10, dim=-1)
            for (k, v) in LOGPROBS_DICT.items()
        }

        for batch_idx in (tqdm(range(BATCH_SIZE), "LOGITS") if progress_bar else range(BATCH_SIZE)):

            html_orig = generate_html_for_logit_plot(
                toks = data_toks,
                logprobs = LOGPROBS_DICT["orig"],
                logprobs_top10 = LOGPROBS_TOP10_DICT["orig"],
                non_ablated_logprobs = LOGPROBS_DICT["orig"],
                batch_idx = batch_idx,
                model = model,
            )
            HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)] = str(html_orig)

            for (layer, head) in negative_heads:
                head_name = f"{layer}.{head}"

                # Save new log probs (post-ablation)
                for effect in ["direct", "indirect", "both"]:
                    for ln_mode in ["frozen", "unfrozen"]:
                        for ablation_mode in ["zero", "mean"]:
                            full_ablation_mode = "+".join([effect, ln_mode, ablation_mode])
                            html_ablated = generate_html_for_logit_plot(
                                toks = data_toks,
                                logprobs = LOGPROBS_DICT[(full_ablation_mode, head_name)],
                                logprobs_top10 = LOGPROBS_TOP10_DICT[(full_ablation_mode, head_name)],
                                non_ablated_logprobs = LOGPROBS_DICT["orig"],
                                batch_idx = batch_idx,
                                model = model,
                            )
                            HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, full_ablation_mode)] = str(html_ablated)

                # Save direct logit effect
                for ln_mode in ["frozen", "unfrozen"]:
                    for ablation_mode in ["zero", "mean"]:
                        full_ablation_mode = "+".join([ln_mode, ablation_mode])
                        dla_neg, dla_pos = generate_html_for_DLA_plot(
                            toks = data_toks[batch_idx],
                            dla_logits = model_results.dla[(ln_mode, ablation_mode)][layer, head][batch_idx],
                            model = model,
                        )
                        HTML_PLOTS["DLA"][(batch_idx, head_name, full_ablation_mode, "neg")] = str(dla_neg)
                        HTML_PLOTS["DLA"][(batch_idx, head_name, full_ablation_mode, "pos")] = str(dla_pos)



    # ! (5) Calculate the attention probs

    if "ATTN" in restrict_computation:

        for batch_idx in (tqdm(range(BATCH_SIZE), "ATTN") if progress_bar else range(BATCH_SIZE)):

            for layer, head in negative_heads:
                head_name = f"{layer}.{head}"

                # Calculate attention, and info-weighted attention
                attn = model_results.pattern[layer, head][batch_idx]
                weighted_attn = einops.einsum(
                    model_results.pattern[layer, head][batch_idx],
                    model_results.out_norm[layer, head][batch_idx] / model_results.out_norm[layer, head][batch_idx].max(),
                    "seqQ seqK, seqK -> seqQ seqK"
                )

                for vis_name, vis_type in {"Large": cv.attention.attention_heads, "Small": cv.attention.attention_patterns}.items():
                    html_standard, html_weighted = [
                        vis_type(
                            attention = x.unsqueeze(0), # (heads=2, seqQ, seqK)
                            tokens = data_str_toks_parsed[batch_idx], # list of length seqQ
                        )
                        for x in [attn, weighted_attn]
                    ]
                    html_standard, html_weighted = list(map(attn_filter, [html_standard, html_weighted]))
                    HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, "standard")] = str(html_standard)
                    HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, "info-weighted")] = str(html_weighted)

            
    # ! (6) Calculate the component of the unembeddings in pre-head residual stream

    if "UNEMBEDDINGS" in restrict_computation:

        for layer, head in negative_heads:
            head_name = f"{layer}.{head}"
            # Get the unembedding components in resid_pre just before this head
            logit_lens: Tensor = model_results.logit_lens[layer]
            logprobs = logit_lens.log_softmax(-1)
            # Get the top 10 logprobs, and get the top logprobs for words which are actually in context
            logprobs_top10 = logprobs.topk(dim=-1, k=10)
            logprobs_top5_in_ctx = get_top_logprobs_in_context(logprobs, data_toks)
            for batch_idx in (tqdm(range(BATCH_SIZE), f"UNEMBEDDINGS: {head_name}") if progress_bar else range(BATCH_SIZE)):
                html = generate_html_for_logit_plot(
                    toks = data_toks,
                    logprobs = logprobs,
                    logprobs_top10 = logprobs_top10,
                    logprobs_top5_in_ctx = logprobs_top5_in_ctx,
                    non_ablated_logprobs = logprobs,
                    batch_idx = batch_idx,
                    model = model,
                )
                HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)] = str(html)

    # Optionally, save the files (we do this if we're generating it from OWT, for the Streamlit page)
    if save_files:
        # If we've only computed a few new things (to save time), then make sure you keep the old values
        if len(restrict_computation) < 4:
            with gzip.open(ST_HTML_PATH / "GZIP_HTML_PLOTS.pkl", "rb") as f:
                HTML_PLOTS_OLD = pickle.load(f)
            for k, v in HTML_PLOTS.items():
                if len(v) > 0:
                    HTML_PLOTS_OLD[k] = v
            HTML_PLOTS = HTML_PLOTS_OLD
        # Now, save the new file
        with gzip.open(ST_HTML_PATH / "GZIP_HTML_PLOTS.pkl", "wb") as f:
            pickle.dump(HTML_PLOTS, f)

    return HTML_PLOTS



def first_occurrence(tensor):
    series = pd.Series(tensor)
    duplicates = series.duplicated(keep='first')
    inverted = ~duplicates
    return inverted.values

def first_occurrence_2d(matrix):
    return t.from_numpy(np.array([first_occurrence(row) for row in matrix]))


def get_top_logprobs_in_context(
    logprobs: Float[Tensor, "batch seqQ d_vocab"],
    toks: Int[Tensor, "batch seqQ"],
):
    '''
    Returns the top predicted logprobs, over all the source tokens in context.

    The indices of the result are the seqK positions, and the values are the logprobs.
    '''
    b, seq, v = logprobs.shape

    # Get all logprobs for the source tokens in context
    b_indices = einops.repeat(t.arange(b), "b -> b sQ sK", sQ=seq, sK=seq)
    sQ_indices = einops.repeat(t.arange(seq), "sQ -> b sQ sK", b=b, sK=seq)
    toks_rep = einops.repeat(toks, "b sK -> b sQ sK", sQ=seq)
    logprobs_ctx: Float[Tensor, "batch seqQ seqK"] = logprobs[b_indices, sQ_indices, toks_rep]
    # The (b, q, k)-th elem is the logprobs of word k at sequence position (b, q)

    # Now we mask wherever q < k, and wherever a token is the first instance of that token (the latter
    # because we want the top 5 DISTINCT words)
    sQ_indices = einops.repeat(t.arange(seq), "sQ -> b sQ 1", b=b)
    sK_indices = einops.repeat(t.arange(seq), "sK -> b 1 sK", b=b)
    causal_mask = sQ_indices >= sK_indices
    first_occurrence_mask = einops.repeat(first_occurrence_2d(toks), "b sK -> b 1 sK")
    logprobs_masked = t.where(
        causal_mask & first_occurrence_mask,
        logprobs_ctx,
        -float("inf")
    )

    # Now, we can pick the top 5 (over the seqK-dimension) for each query index
    k = min(5, logprobs_masked.size(-1))
    logprobs_masked_top5 = logprobs_masked.topk(dim=-1, k=k)

    return logprobs_masked_top5


