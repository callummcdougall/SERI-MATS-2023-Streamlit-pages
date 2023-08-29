# This contains functions to generate HTML for each of the different Streamlit visualisations.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
import gzip
from typing import Tuple, List, Any, Optional, Dict
import torch as t
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from IPython.display import display, HTML
from plotly.colors import sample_colorscale
import einops
import circuitsvis as cv
import pickle
import pandas as pd
import numpy as np
import time
import itertools

from transformer_lens.rs.callum2.utils import (
    ST_HTML_PATH,
    NEGATIVE_HEADS,
    first_occurrence_2d
)
from transformer_lens.rs.callum2.generate_st_html.model_results import (
    ModelResults,
    get_model_results
)
from transformer_lens.rs.callum2.cspa.cspa_functions import (
    FUNCTION_STR_TOKS,
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
    # designed to reduce the space taken up by attention plots, by only showing to 4dp
    html = str(html)
    def round_match(match):
        return "{:.4f}".format(float(match.group()))
    return re.sub(r'\b0\.\d+\b', round_match, html)


def rearrange_list(my_list, list_length):
    assert len(my_list) % list_length == 0
    return [my_list[i:i+list_length] for i in range(0, len(my_list), list_length)]


def _get_color(importances):
    """
    Returns a color based on the importance of a word.

    Also returns color for the text (has to be white if sufficiently extreme).    
    """
    importances_filtered = [i if (i is not None) else 0.5 for i in importances]
    bg_colors = sample_colorscale("RdBu", importances_filtered, low=0.0, high=1.0, colortype='rgb')
    text_colors = list(map(lambda i: "white" if abs(i - 0.5) > 0.31 else "black", importances_filtered))
    for i, imp in enumerate(importances):
        if imp is None:
            bg_colors[i] = "white"
            text_colors[i] = "grey"
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
    assert len(words) == len(importances), "Words and importances must be of same length"
    assert len(words) == len(hover_text_list), "Words and hover_text_list must be of same length"

    tags = ["<td>"]
    bg_colors, text_colors = _get_color(importances)
    for word, bg_color, text_color, hover_text in zip(words, bg_colors, text_colors, hover_text_list):
        word = word.replace(" ", "&nbsp;")
        if "\n" in word:
            word = "\\n"
        unwrapped_tag = f'<mark style="background-color:{bg_color}"><font color="{text_color}">{word}</font></mark>'
        unwrapped_tag = f'<span class="tooltip">{unwrapped_tag}TOOLTIPTEXT</span>'
        unwrapped_tag = unwrapped_tag.replace("TOOLTIPTEXT", f'<span class="tooltiptext">{hover_text}</span>' if hover_text else "")
        tags.append(unwrapped_tag)
    tags.append("</td>")
    html = "&nbsp;".join(tags)
    return html



def clip(x):
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 1.0
    return x


def generate_html_for_cspa_plots(
    str_toks_list: List[List[str]],
    cspa_results: Dict[str, Float[Tensor, "batch seq-1"]],
    s_sstar_pairs: Dict[Tuple[int, int], List[Tuple[float, int, int]]],
    cutoff: float = 0.03,
):
    '''
    Anything that's not in the top 5% (in either extreme) we don't bother coloring.

    If it's in the top 5% of loss-decreasing examples (i.e. projected - orig loss is positive and large), then we color
    it blue in situations where this isn't captured by CSPA (i.e. projected_diff / ablated_diff is close to 1).
    '''
    loss_ablated_minus_orig = cspa_results["loss_ablated"] - cspa_results["loss"]
    loss_cspa_minus_orig = cspa_results["loss_cspa"] - cspa_results["loss"]

    n_tokens = sum(map(len, str_toks_list))
    if n_tokens > 1000:
        top_5pct = t.quantile(loss_ablated_minus_orig, 1 - cutoff).item()
        bottom_5pct = t.quantile(loss_ablated_minus_orig, cutoff).item()
    else:
        top_5pct = 0.14
        bottom_5pct = -0.14
    
    html_plots = {}

    for b, (str_toks, l_origs, l_abl_diffs, l_proj_diffs) in enumerate(zip(str_toks_list, cspa_results["loss"], loss_ablated_minus_orig, loss_cspa_minus_orig)):
        
        importances = [None] * len(str_toks)
        hover_text_list = []

        for sQ, (l_orig, l_abl_diff, l_proj_diff) in enumerate(zip(l_origs, l_abl_diffs, l_proj_diffs)):
            # Get ratio of projected diff over ablated diff (this will usually be positive, and it'll be close to zero when our model is good)
            ratio = (l_proj_diff / l_abl_diff).item()

            hover_text = ""

            if (l_abl_diff > top_5pct) or (l_abl_diff < bottom_5pct):
                # Make this token bold
                str_toks[sQ] = "<b>" + str_toks[sQ] + "</b>"
                # Importance is higher when our model doesn't capture what's going on here
                importances[sQ] = clip(0.5 + (1 if (l_abl_diff > top_5pct) else -1) * ((1 - ratio) / 2))
                # Add the basic loss diff hover text
                hover_text += f"Orig loss: {l_orig:.3f}<br>Δ loss from ablation: {l_abl_diff:.3f}<br>Δ loss from CSPA: {l_proj_diff:.3f}"
                # Add the hover text from (s, s*) pairs
                table_body = "".join([
                    f"<tr><td>{s}</td><td>{sstar}</td><td>{LL:.2f}</td></tr>"
                    for LL, s, sstar in s_sstar_pairs[(b, sQ)]
                ])
                hover_text += "<br><br>Info moved by CSPA:<br>"
                hover_text += f"<table><thead><tr><th>s</th><th>s*</th><th>Logit Lens</th></tr></thead><tbody>{table_body}</tbody></table>"
            
            hover_text_list.append(hover_text)
        
        hover_text_list.append("(next token unknown)")

        html = format_word_importances(str_toks, importances, hover_text_list)
        html_plots[(b,)] = html
    
    return html_plots



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

    hover_text_list = [f"({i}) <b>'{s}'</b><br>{d:.4f}" for i, (s, d) in enumerate(zip(str_toks[:-1], loss_diff))] + ["(next token unknown)"]

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
    full_toks: Int[Tensor, "batch seq"],
    full_logprobs: Float[Tensor, "batch seq d_vocab"],
    full_non_ablated_logprobs: Float[Tensor, "batch seq d_vocab"],
    model: HookedTransformer,
    full_logprobs_top5_in_ctx: Any = None,
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
    batch_size, seq_len = full_toks.shape

    max_color = 2.5

    # Get the top10 logprobs, and the corresponding str toks
    full_logprobs_top10 = full_logprobs.topk(10, dim=-1)
    full_str_toks_top10 = [rearrange_list(model.to_str_tokens(logprobs_top10_indices.flatten()), 10) for logprobs_top10_indices in full_logprobs_top10.indices]

    # Get the thing I'll use to color the tokens (leaving white if there's basically no difference)
    all_str_toks = rearrange_list(model.to_str_tokens(full_toks.flatten()), list_length=seq_len)
    batch_idx = einops.repeat(t.arange(batch_size), 'b -> b seq', seq=seq_len-1)
    seq_idx = einops.repeat(t.arange(seq_len-1), 'seq -> b seq', b=batch_size)
    logprobs_on_correct_token = full_logprobs[batch_idx, seq_idx, full_toks[:, 1:]]
    logprobs_on_correct_token_baseline = full_non_ablated_logprobs[batch_idx, seq_idx, full_toks[:, 1:]]
    probs_on_correct_token = logprobs_on_correct_token.exp()
    all_colors = []
    for L, L_baseline in zip(logprobs_on_correct_token, logprobs_on_correct_token_baseline):
        if (L - L_baseline).abs().max() < 1e-4:
            # In this case, we must have the original logprobs, so we set colors based on just these
            orig = True
            # 0.5 is the baseline for "very low probability" (logprob=-4), we get 1.0 close to logprob=0
            colors = 1 + t.maximum(L / (2 * max_color), t.full_like(L, -0.5))
        else:
            # In this case, these must be ablated logprobs, so we set colors based on the difference
            orig = False
            # 0.5 is the baseline for "no change in logprob", we get 1.0 or 0.0 for a change of ±4
            colors = t.clip(0.5 + (L - L_baseline) / (2 * max_color), 0.0, 1.0)
        all_colors.append(colors.tolist() + [0.5])

    # Now, get the top10 predictions for each token (note that `full_logprobs_top5_in_ctx` are indices over seq_pos, not over d_vocab)
    if full_logprobs_top5_in_ctx is not None:
        max_color = 30
        seqpos_top5_ctx = full_logprobs_top5_in_ctx.indices # shape (batch, seqQ, 5)
        batch_idx = einops.repeat(t.arange(batch_size), 'b -> b seq k', seq=seq_len, k=seqpos_top5_ctx.shape[-1])
        tokenIDs_top5_ctx = full_toks[batch_idx, seqpos_top5_ctx] # shape (batch, seqQ, 5)
        str_toks_top5_ctx_all = [rearrange_list(model.to_str_tokens(tokenIDs.flatten()), 5) for tokenIDs in tokenIDs_top5_ctx]

    # Now, get the hovertext for my formatting function


    logprobs_on_correct_token_rep = einops.repeat(logprobs_on_correct_token, "b seqQ -> b seqQ 1")
    all_correct_ranks = (logprobs_on_correct_token_rep < full_logprobs[:, :seq_len-1]).sum(dim=-1)
    if full_logprobs_top5_in_ctx is not None:
        logprobs_top5_ctx_all_rep = einops.repeat(full_logprobs_top5_in_ctx.values, 'b seqQ K -> b seqQ 1 K')
        logprobs_rep = einops.repeat(full_logprobs, 'b seqQ d_vocab -> b seqQ d_vocab 1')
        all_top5_ctx_ranks: Int[Tensor, "b seqQ K"] = (logprobs_top5_ctx_all_rep < logprobs_rep).sum(dim=-2)

    html_list = []

    for batch_idx in range(full_toks.size(0)):
        hover_text_list = []

        for seq_pos in range(seq_len-1):

            current_word = all_str_toks[batch_idx][seq_pos]
            next_word = all_str_toks[batch_idx][seq_pos + 1]

            correct_rank = all_correct_ranks[batch_idx, seq_pos]
            str_toks_top10 = full_str_toks_top10[batch_idx][seq_pos]
            logprobs_top10 = full_logprobs_top10.values[batch_idx, seq_pos]
            probs_top10 = logprobs_top10.exp()
            if full_logprobs_top5_in_ctx is not None:
                str_toks_top5_ctx = str_toks_top5_ctx_all[batch_idx][seq_pos]
                logprobs_top5_ctx = full_logprobs_top5_in_ctx.values[batch_idx, seq_pos]
                probs_top5_ctx = logprobs_top5_ctx.exp()
                top5_ctx_ranks = all_top5_ctx_ranks[batch_idx, seq_pos]

            table_body = ""
            for idx, (word, logprob, prob) in enumerate(zip(str_toks_top10, logprobs_top10, probs_top10)):
                table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{logprob:.2f}</td><td>{prob:.2%}</td></tr>"

            lp_orig = logprobs_on_correct_token_baseline[batch_idx, seq_pos]
            lp = logprobs_on_correct_token[batch_idx, seq_pos]
            p = probs_on_correct_token[batch_idx, seq_pos]
            empty_row = '<tr class="empty-row"><td></td><td></td><td></td><td></td></tr>'
            new_hover_text = "".join([
                "<table>",
                "<thead><tr><th>Rank</th><th>Word</th><th>Logprob</th><th>Prob</th></tr></thead>",
                f"<tbody><tr><td>#{correct_rank}</td><td>{next_word!r}</td><td>{lp:.2f}</td><td>{p:.2%}</td></tr>"
            ]).replace('<td>', '<td style="background-color:#c2d9ff; color:black">') + f"{empty_row}{table_body}</tbody></table>"
            if not(orig):
                new_hover_text = f"Δ logprob on correct token = {lp-lp_orig:.2f}<br><br>" + new_hover_text

            # Finally, get the top5 in context thing, if we are given it
            if full_logprobs_top5_in_ctx is not None:
                logprobs_mean = full_logprobs[batch_idx, seq_pos].mean()
                table_body = ""
                for (rank, word, logprob, prob) in zip(top5_ctx_ranks, str_toks_top5_ctx, logprobs_top5_ctx, probs_top5_ctx):
                    if logprob > -1e4:
                        # * Not coloring in this way because it's visually unclear and confusing (e.g. function words, what baseline to use).
                        # colors[seq_pos] = min(1, 0.5 + (logprob - logprobs_mean).item() / (2 * max_color))
                        table_body += f"<tr><td>#{rank}</td><td>{word!r}</td><td>{logprob:.2f}</td><td>{prob:.2%}</td></tr>"
                new_hover_text += f"<br><b>Top 5 non-fn word predictions from context:</b><br><br>" + "".join([
                    "<table>",
                    "<thead><tr><th>Rank</th><th>Word</th><th>Logprob</th><th>Prob</th></tr></thead>",
                    f"<tbody>{table_body}</tbody></table>"
                ])

            hover_text_list.append(f"<span background-color:'#ddd'>{current_word!r}</span> ➔ <span background-color:'#ddd'>{next_word!r}</span><br><br>{new_hover_text}")

        html = format_word_importances(all_str_toks[batch_idx], all_colors[batch_idx], hover_text_list + ["(next token unknown)"])
        html_list.append(html)

    return html_list





def generate_html_for_DLA_plot(
    toks: Int[Tensor, "batch seq_len"],
    dla_logits: Float[Tensor, "batch seq_len-1 d_vocab"],
    model: HookedTransformer,
):
    batch_size, seq_len = toks.shape
    dla_logits_topk = dla_logits.topk(dim=-1, k=10)
    dla_logits_botk = dla_logits.topk(dim=-1, k=10, largest=False)

    # * From eyeballing data, it looks like logit suppression of 2.5 is a sufficiently extreme value to warrant setting this as the default upper color
    all_importances_pos_raw: Float[Tensor, "b s-1"] = dla_logits.max(dim=-1).values
    all_importances_neg_raw: Float[Tensor, "b s-1"] = (-dla_logits).max(dim=-1).values
    all_importances = {"pos": [], "neg": []}
    for importances_pos_raw, importances_neg_raw in zip(all_importances_pos_raw, all_importances_neg_raw):
        denom = 2 * max(2.5, importances_pos_raw.max().item(), importances_neg_raw.max().item())
        all_importances["pos"].append((0.5 + (importances_pos_raw / denom)).tolist())
        all_importances["neg"].append((0.5 - (importances_neg_raw / denom)).tolist())

    words = rearrange_list(model.to_str_tokens(toks.flatten()), seq_len)
    words_nbsp = [list(map(lambda word: word.replace(" ", "&nbsp;"), words_list)) + ["(can't see next token)"] for words_list in words]
    words = [words_list + ["(can't see next token)"] for words_list in words]

    all_hover_text = {"pos": [], "neg": []}

    for b in range(batch_size):
        hover_text_list_pos = []
        hover_text_list_neg = []

        for s in range(seq_len):
            current_word = words_nbsp[b][s]
            next_word = words_nbsp[b][s+1]

            top10_values = dla_logits_topk.values[b, s].tolist()
            top10_str_toks = model.to_str_tokens(dla_logits_topk.indices[b, s])
            bot10_values = dla_logits_botk.values[b, s].tolist()
            bot10_str_toks = model.to_str_tokens(dla_logits_botk.indices[b, s])

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
        all_hover_text["pos"].append(hover_text_list_pos)
        all_hover_text["neg"].append(hover_text_list_neg)

    html_pos_list = [format_word_importances(words[b][:-1], all_importances["pos"][b], all_hover_text["pos"][b]) for b in range(batch_size)]
    html_neg_list = [format_word_importances(words[b][:-1], all_importances["neg"][b], all_hover_text["neg"][b]) for b in range(batch_size)]

    return html_pos_list, html_neg_list




def generate_4_html_plots(
    model: HookedTransformer,
    data_toks: Float[Tensor, "batch seq_len"],
    data_str_toks_parsed: List[List[str]],
    negative_heads: List[Tuple[int, int]],
    save_files: bool,
    model_results: Optional[ModelResults],
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]],
    verbose: bool = False,
    restrict_computation: List[str] = ["LOSS", "LOGITS", "ATTN", "UNEMBEDDINGS"],
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
        model_results = get_model_results(model, data_toks, negative_heads=negative_heads, result_mean=result_mean, verbose=verbose)


    # ! Calculate the loss diffs from ablating

    if "LOSS" in restrict_computation:

        if verbose: print(f"{'LOSS':<12} ... ", end="\r"); t0 = time.time()
        
        for batch_idx in range(BATCH_SIZE):
            for (layer, head) in negative_heads:
                head_name = f"{layer}.{head}"
                # For each different type of ablation, get the loss diffs
                for full_ablation_type_tuple, loss_diff_by_head in model_results.loss_diffs.items():
                    # Get the loss per token, and pad with zeros at the end (cause we don't know the last value!)
                    full_ablation_type_name = "+".join(full_ablation_type_tuple)
                    if not((layer == 11) and ("excluding 11.10" in full_ablation_type_name)):
                        loss_diff = loss_diff_by_head[layer, head][batch_idx]
                        loss_diff_padded = t.cat([loss_diff, t.zeros((1,))], dim=-1)
                        html_25, html_max = generate_html_for_loss_plot(
                            data_str_toks_parsed[batch_idx],
                            loss_diff = loss_diff_padded,
                        )
                        HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_type_name, True)] = str(html_max)
                        HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_type_name, False)] = str(html_25)

        if verbose: print(f"{'LOSS':<12} ... {time.time()-t0:.2f}s")


    # ! Calculate the logits & direct logit attributions
    # For logprobs, we batch this - first calculating all the logprobs and the top10s, then passing them all into the `generate_html_for_logit_plot` 
    # function along with a `batch_idx` argument.
    # For DLA, we don't batch it, i.e. we just pass in the vector of DLA logits.

    if "LOGITS" in restrict_computation:

        if verbose: print(f"{'LOGITS ORIG':<12} ... ", end="\r"); t0 = time.time()
        logprobs_orig = model_results.logits_orig.log_softmax(-1)
        html_orig_list = generate_html_for_logit_plot(
            full_toks = data_toks,
            full_logprobs = logprobs_orig,
            full_non_ablated_logprobs = logprobs_orig,
            model = model,
        )
        for batch_idx, html_orig in enumerate(html_orig_list):
            HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)] = str(html_orig)
        if verbose: print(f"{'LOGITS ORIG':<12} ... {time.time()-t0:.2f}s")

        for (layer, head) in negative_heads:
            head_name = f"{layer}.{head}"
            if verbose: print(f"{'LOGITS ' + head_name:<12} ... ", end="\r"); t0 = time.time()

            # Save new log probs (post-ablation)
            for full_ablation_type, logits_by_head in model_results.logits.items():
                full_ablation_type_name = "+".join(full_ablation_type)
                if head_name not in full_ablation_type_name: # this makes sure "indirect excluding 11.10" isn't counted for head 11.10
                    html_list = generate_html_for_logit_plot(
                        full_toks = data_toks,
                        full_logprobs = logits_by_head[layer, head].log_softmax(-1),
                        full_non_ablated_logprobs = logprobs_orig,
                        model = model,
                    )
                    for batch_idx, html in enumerate(html_list):
                        HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, full_ablation_type_name)] = str(html)

            # Save direct logit attribution
            for ln_mode, ablation_mode in itertools.product(["frozen", "unfrozen"], ["mean"]): # ["zero", "mean"]
                full_ablation_mode = "+".join([ln_mode, ablation_mode])
                html_pos_list, html_neg_list = generate_html_for_DLA_plot(
                    toks = data_toks,
                    dla_logits = model_results.dla[(ln_mode, ablation_mode)][layer, head],
                    model = model,
                )
                for batch_idx, (html_pos, html_neg) in enumerate(zip(html_pos_list, html_neg_list)):
                    HTML_PLOTS["DLA"][(batch_idx, head_name, full_ablation_mode, "neg")] = str(html_neg)
                    HTML_PLOTS["DLA"][(batch_idx, head_name, full_ablation_mode, "pos")] = str(html_pos)
            
            if verbose: print(f"{'LOGITS ' + head_name:<12} ... {time.time()-t0:.2f}s")


    # ! Calculate the attention probs

    if "ATTN" in restrict_computation:

        if verbose: print(f"{'ATTN':<12} ... ", end="\r"); t0 = time.time()

        for batch_idx in range(BATCH_SIZE):

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
            
        if verbose: print(f"{'ATTN':<12} ... {time.time()-t0:.2f}s")

            
    # ! Calculate the component of the unembeddings in pre-head residual stream

    FUNCTION_TOKS = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze()

    if "UNEMBEDDINGS" in restrict_computation:

        if verbose: print(f"{'UNEMBEDDINGS':<12} ... ", end="\r"); t0 = time.time()

        for layer, head in negative_heads:
            head_name = f"{layer}.{head}"

            # Get the unembedding components in resid_pre just before this head
            logit_lens: Tensor = model_results.logit_lens[layer]
            logprobs = logit_lens.log_softmax(-1)

            html_list = generate_html_for_logit_plot(
                full_toks = data_toks,
                full_logprobs = logprobs,
                full_logprobs_top5_in_ctx = get_top_logprobs_in_context(logprobs, data_toks, FUNCTION_TOKS),
                full_non_ablated_logprobs = logprobs,
                model = model,
            )
            for batch_idx, html in enumerate(html_list):
                HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)] = str(html)

        if verbose: print(f"{'UNEMBEDDINGS':<12} ... {time.time()-t0:.2f}s\n")


    # Optionally, save the files (we do this if we're generating it from OWT, for the Streamlit page)
    if save_files:
        if verbose: print(f"{'Saving':<12} ... ", end="\r"); t0 = time.time()
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
        if verbose: print(f"{'Saving':<12} ... {time.time()-t0:.2f}s")

    return HTML_PLOTS






def get_top_logprobs_in_context(
    logprobs: Float[Tensor, "batch seqQ d_vocab"],
    toks: Int[Tensor, "batch seqQ"],
    function_toks: Int[Tensor, "toks"]
):
    '''
    Returns the top predicted logprobs, over all the source tokens in context.

    The indices of the result are the seqK positions, and the values are the logprobs.

    Tokens which are function words are filtered out.
    '''
    b, seq, v = logprobs.shape

    # Get all logprobs for the source tokens in context
    b_indices = einops.repeat(t.arange(b), "b -> b sQ sK", sQ=seq, sK=seq)
    sQ_indices = einops.repeat(t.arange(seq), "sQ -> b sQ sK", b=b, sK=seq)
    toks_rep = einops.repeat(toks, "b sK -> b sQ sK", sQ=seq)
    logprobs_ctx: Float[Tensor, "batch seqQ seqK"] = logprobs[b_indices, sQ_indices, toks_rep]
    # The (b, q, k)-th elem is the logprobs of word k at sequence position (b, q)

    # Mask: causal
    sQ_indices = einops.repeat(t.arange(seq), "sQ -> b sQ 1", b=b)
    sK_indices = einops.repeat(t.arange(seq), "sK -> b 1 sK", b=b)
    causal_mask = sQ_indices >= sK_indices
    # Mask: first occurrence of each word (because we want the top 5 DISTINCT words)
    first_occurrence_mask = einops.repeat(first_occurrence_2d(toks), "b sK -> b 1 sK")
    # Mask: non-function words
    non_fn_word_mask = (toks[:, :, None] != function_toks).all(dim=-1)
    non_fn_word_mask = einops.repeat(non_fn_word_mask, "b sK -> b 1 sK")
    # Apply all 3 masks
    logprobs_masked = t.where(
        causal_mask & first_occurrence_mask & non_fn_word_mask,
        logprobs_ctx,
        -float("inf")
    )

    # Now, we can pick the top 5 (over the seqK-dimension) for each query index
    k = min(5, logprobs_masked.size(-1))
    logprobs_masked_top5 = logprobs_masked.topk(dim=-1, k=k)

    return logprobs_masked_top5




def generate_4_html_plots_batched(
    model: HookedTransformer,
    data_toks: Float[Int, "batch seq_len"],
    data_str_toks_parsed: List[List[str]],
    max_batch_size: int = 50,
    start_idx: int = 0,
    negative_heads: List[Tuple[int, int]] = NEGATIVE_HEADS,
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[Tuple, str]]:

    # Generate all HTML plots, and save them
    # (this is important, in case we get a CPU error half way through)
    print("Generating HTML plots...\n")
    if max_batch_size > data_toks.shape[0]:
        max_batch_size = data_toks.shape[0]
    chunks = data_toks.shape[0] // max_batch_size
    lower_upper_list = []
    for i, _toks in enumerate(t.chunk(data_toks, chunks=chunks)):
        lower, upper = i*max_batch_size, (i+1)*max_batch_size
        lower_upper_list.append((lower, upper))
        if lower >= start_idx:
            html_plots = generate_4_html_plots(
                model=model,
                data_toks=_toks,
                data_str_toks_parsed=data_str_toks_parsed[lower: upper],
                negative_heads=negative_heads,
                result_mean=result_mean,
                verbose=verbose,
                save_files=False,
                model_results=None,
            )
            with gzip.open(ST_HTML_PATH / f"_GZIP_HTML_PLOTS_{lower}_{upper}.pkl", "wb") as f:
                pickle.dump(html_plots, f)
    
    # Gather all the HTML plots, one by one
    print("Gathering HTML plots...")
    html_plots = None
    for (lower, upper) in lower_upper_list:
        with gzip.open(ST_HTML_PATH / f"_GZIP_HTML_PLOTS_{lower}_{upper}.pkl", "rb") as f:
            html_plots_new = pickle.load(f)
        html_plots = update_html_plots(html_plots, html_plots_new)
    # Save them
    B, S = data_toks.shape
    filename = ST_HTML_PATH / f"GZIP_HTML_PLOTS_b{B}_s{S}.pkl"
    print(f"Saving HTML plots as a single dict, at '{filename.name}'...")
    with gzip.open(filename, "wb") as f:
        pickle.dump(html_plots, f)
    # Delete the plots we no longer need
    print("Deleting HTML plots we no longer need...")
    for (lower, upper) in lower_upper_list:
        file = ST_HTML_PATH / f"_GZIP_HTML_PLOTS_{lower}_{upper}.pkl"
        if file.exists(): os.remove(file)



def update_html_plots(
    html_plots: Optional[dict],
    html_plots_new: dict,
):
    '''
    If html_plots and html_plots_new both have the same batch indices, then we need
    to be careful merging them, because we want to increment all the batch indices in
    the new plots dict by the appropriate amount. This function does that.
    '''
    if html_plots is None: return html_plots_new

    # Find the batch index
    dla_keys = html_plots["DLA"].keys()
    start_batch_idx = max([k[0] for k in dla_keys]) + 1

    for html_plot_type, dict_of_html_plots in html_plots_new.items():
        for (batch_idx, *other_keys), html_plot in dict_of_html_plots.items():
            new_key = (batch_idx + start_batch_idx, *other_keys)
            html_plots[html_plot_type][new_key] = html_plot
    
    return html_plots