# This contains functions to generate HTML for each of the different Streamlit visualisations.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
try:
    root_dir = os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts"
    os.chdir(root_dir)
except:
    root_dir = "/app/seri-mats-2023-streamlit-pages/explore_prompts"
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

from explore_prompts_utils import (
    ST_HTML_PATH,
    NEGATIVE_HEADS
)
from model_results import (
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
    line-height: 1.48em;
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

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>

<script>
$(document).ready(function(){
  $('.tooltip').hover(function(){
    var tooltipWidth = $(this).children('.tooltiptext').outerWidth();
    var viewportWidth = $(window).width();
    var tooltipRight = $(this).offset().left + tooltipWidth;
    if (tooltipRight > viewportWidth) {
      $(this).children('.tooltiptext').css('left', 'auto').css('right', '0');
    }
  }, function() {
    $(this).children('.tooltiptext').css('left', '50%').css('right', 'auto');
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




def _get_color(importance):
    """
    Returns a color based on the importance of a word.

    Also returns color for the text (has to be white if sufficiently extreme).    
    """
    bg_color = sample_colorscale("RdBu", importance, low=0.0, high=1.0, colortype='rgb')[0]
    text_color = "white" if abs(importance - 0.5) > 0.3 else "black"
    return bg_color, text_color



def format_word_importances(
    words: List[str],
    importances: List[float],
    hover_text_list: List[str],
    word_gaps: bool = True,
    show: bool = False,
    save: bool = False,
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
    for word, importance, hover_text in zip(words, importances, hover_text_list):
        word = word.replace(" ", "&nbsp;")
        bg_color, text_color = _get_color(importance)
        unwrapped_tag = f'<mark style="background-color:{bg_color};opacity:1.0;line-height:{"1.75em" if word_gaps else "1.48em"}"><font color="{text_color}">{word}</font></mark>'
        unwrapped_tag = f'<span class="tooltip">{unwrapped_tag}<span class="tooltiptext">{hover_text}</span></span>'
        tags.append(unwrapped_tag)
    tags.append("</td>")
    html = ("&nbsp;" if word_gaps else "").join(tags)

    if show:
        display(HTML(CSS + "<br>" * 10 + html))
    elif save:
        with open("file.html", "w") as f:
            f.write("<br>" * 10 + CSS + html)
    else:
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
    max_color = 2.5

    importances = t.clip(0.5 + loss_diff / (2 * max_color), 0.0, 1.0).tolist()

    hover_text_list = [f"<b>'{s}'</b><br>{d:.4f}" for (s, d) in zip(str_toks[:-1], loss_diff)] + [""]

    html = format_word_importances(str_toks, importances, hover_text_list)
    return html



def generate_html_for_unembedding_components_plot(
    str_toks: List[List[str]], # shape (batch, seq) really
    unembedding_components_avg: Float[Tensor, "batch seq"],
    unembedding_components_top10, # topk return type, shape (batch, seq, 10)
):
    results = {}

    for batch_idx in range(len(str_toks)):

        importances = []
        hover_text_list = []

        for seq_pos in range(unembedding_components_avg.size(1)):
            avg = unembedding_components_avg[batch_idx, seq_pos].item()
            top_values = unembedding_components_top10.values[batch_idx, seq_pos, :].tolist()
            top_indices = unembedding_components_top10.indices[batch_idx, seq_pos, :].tolist()
            top_words = [str_toks[batch_idx][idx] for idx in top_indices]

            table_body = ""
            for idx, (word, value) in enumerate(zip(top_words, top_values)):
                if value < -1e8: break
                table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{value:.2f}</td></tr>"
                if idx == 0: importances.append(value-avg)
            if table_body == "":
                importances.append(0.0)

            empty_row = '<tr class="empty-row"><td></td><td></td><td></td></tr>'
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
):
    '''
    This gets the top predicted tokens for the model (and the prediction for the actual next token).

    We use non_ablated_logits as a baseline. The baseline tokens will be colored by the logprob of the
    token which actually came next, the non-baseline tokens will be colored by the difference in this prob
    from the baseline to them.
    '''
    toks = toks[batch_idx]
    logprobs = logprobs[batch_idx]
    non_ablated_logprobs = non_ablated_logprobs[batch_idx]

    max_color = 4 # This is the fixed point for the color scale

    # Get the thing I'll use to color the tokens (leaving white if there's basically no difference)
    str_toks = model.to_str_tokens(toks)
    logprobs_on_correct_token = logprobs[range(len(str_toks)-1), toks[1:]]
    logprobs_on_correct_token_baseline = non_ablated_logprobs[range(len(str_toks)-1), toks[1:]]
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
    str_toks_top10_all = [model.to_str_tokens(logprobs_top10.indices[batch_idx, i]) for i in range(logprobs.size(0))]
    logprobs_top10_all = logprobs_top10.values[batch_idx].tolist()

    # Now, get the hovertext for my formatting function

    hover_text_list = []

    for seq_pos in range(len(str_toks) - 1):

        current_word = str_toks[seq_pos]
        next_word = str_toks[seq_pos + 1]

        correct_idx = (logprobs_on_correct_token[seq_pos] < logprobs[seq_pos]).sum()
        str_toks_top10 = str_toks_top10_all[seq_pos]
        logprobs_top10 = t.tensor(logprobs_top10_all[seq_pos])

        table_body = ""
        for idx, (word, value) in enumerate(zip(str_toks_top10, logprobs_top10)):
            table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{value:.2f}</td><td>{value.exp():.2%}</td></tr>"

        lp_orig = logprobs_on_correct_token_baseline[seq_pos]
        lp = logprobs_on_correct_token[seq_pos]
        p = lp.exp()
        empty_row = '<tr class="empty-row"><td></td><td></td><td></td><td></td></tr>'
        new_hover_text = "".join([
            "<table>",
            "<thead><tr><th>Rank</th><th>Word</th><th>Logprob</th><th>Prob</th></tr></thead>",
            f"<tbody><tr><td>#{correct_idx}</td><td>{next_word!r}</td><td>{lp:.2f}</td><td>{p:.2%}</td></tr>"
        ]).replace('<td>', '<td style="background-color:#c2d9ff; color:black">') + f"{empty_row}{table_body}</tbody></table>"
        if not(orig):
            new_hover_text = f"Δ logprob on correct token = {lp-lp_orig:.2f}<br><br>" + new_hover_text
        new_hover_text = f"<span background-color:'#ddd'>{current_word!r}</span> ➔ <span background-color:'#ddd'>{next_word!r}</span><br><br>" + new_hover_text
        hover_text_list.append(new_hover_text)

    hover_text_list = hover_text_list + ["(next token unknown)"]
    html = format_word_importances(str_toks, colors, hover_text_list)

    return html



def generate_html_for_DLA_plot(
    toks: Int[Tensor, "seq_len"],
    dla_logits: Float[Tensor, "seq_len-1 d_vocab"],
    model: HookedTransformer,
):
    dla_logits = dla_logits - dla_logits.mean(-1, keepdim=True)

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
        bot5_values = dla_logits_botk.values[i].tolist()
        bot5_indices = dla_logits_botk.indices[i].tolist()
        bot5_str_toks = list(map(model.to_single_str_token, bot5_indices))

        neg_table_body = ""
        pos_table_body = ""
        for idx, (word, value) in enumerate(zip(bot5_str_toks, bot5_values)):
            neg_table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{value:.2f}</td></tr>"
        for idx, (word, value) in enumerate(zip(top10_str_toks, top10_values)):
            pos_table_body += f"<tr><td>#{idx}</td><td>{word!r}</td><td>{value:.2f}</td></tr>"

        hover_text_list_neg.append("".join([
            f"<span background-color:'#ddd'>{current_word!r}</span> ➔ <span background-color:'#ddd'>{next_word!r}</span><br><br>",
            "<table>",
            "<thead><tr><th>Rank</th><th>Word</th><th>Logit (subtract mean)</th></tr></thead>",
            f"<tbody>{neg_table_body}</tbody></table>",
        ]))
        hover_text_list_pos.append("".join([
            f"<span background-color:'#ddd'>{current_word!r}</span> ➔ <span background-color:'#ddd'>{next_word!r}</span><br><br>",
            "<table>",
            "<thead><tr><th>Rank</th><th>Word</th><th>Logit (subtract mean)</th></tr></thead>",
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
                html = generate_html_for_loss_plot(
                    data_str_toks_parsed[batch_idx],
                    loss_diff = loss_diff[batch_idx],
                )
                HTML_PLOTS["LOSS"][(batch_idx, head_name, ablation_type)] = str(html)
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