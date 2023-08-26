from typing import Dict, List, Tuple, Optional, Literal
import time
import gzip
from jaxtyping import Float, Int
import torch as t
from torch import Tensor
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from transformer_lens import utils, HookedTransformer
from copy import copy
from IPython.display import display, HTML
import pickle
from pathlib import Path

from transformer_lens.rs.callum2.generate_st_html.generate_html_funcs import (
    generate_html_for_cspa_plots,
    generate_html_for_logit_plot,
    generate_html_for_DLA_plot,
    CSS,
)
from transformer_lens.rs.callum2.utils import (
    ST_HTML_PATH,
)

def generate_loss_based_scatter(
    cspa_results,
    nbins: int = 200,
    values: Literal["kl-div", "kl-div-reversed", "loss", "loss-absolute"] = "loss",
):
    xlabels = {
        "kl-div": "D<sub>MA</sub>",
        "kl-div-reversed": "D<sub>KL</sub>",
        "loss": "Change in loss under mean ablation",
        "loss-absolute": "Absolute change in loss under mean ablation",
    }
    ylabels = {
        "kl-div": "D<sub>CSPA</sub>",
        "kl-div-reversed": "D<sub>CSPA</sub>",
        "loss": "Change in loss under CSPA",
        "loss-absolute": "Absolute change in loss under CSPA",
    }  

    assert values in ["kl-div", "kl-div-reversed", "loss", "loss-absolute"]
    if values.startswith("kl-div"):
        xaxis_values = cspa_results["kl_div_ablated_to_orig"].flatten()
        yaxis_values = cspa_results["kl_div_cspa_to_orig"].flatten()
        title = "KL divergence of ablated predictions, relative to clean predictions"
    elif values.startswith("loss"):
        l_orig = cspa_results["loss"].flatten()
        l_abl = cspa_results["loss_ablated"].flatten()
        l_cspa = cspa_results["loss_cspa"].flatten()

        title = "Change in loss from ablation (relative to clean model)"
        xaxis_values = l_abl - l_orig
        yaxis_values = l_cspa - l_orig

        if values == "loss-absolute":
            xaxis_values = xaxis_values.abs()
            yaxis_values = yaxis_values.abs()
            title = "Absolute change in loss from ablation"

    ranks = t.argsort(t.argsort(xaxis_values))
    quantiles = ((ranks.float() / (xaxis_values.numel() - 1)) * nbins).int()
    quantiles = t.clamp(quantiles, 0, nbins-1)

    xaxis_values_per_bucket = []
    yaxis_values_per_bucket = []

    for i in range(0, nbins-1):
        is_in_bucket = (quantiles == i)

        xaxis_value = xaxis_values[is_in_bucket].mean().item()
        yaxis_value = yaxis_values[is_in_bucket].mean().item()

        xaxis_values_per_bucket.append(xaxis_value)
        yaxis_values_per_bucket.append(yaxis_value)

    df = pd.DataFrame({
        xlabels[values]: xaxis_values_per_bucket,
        ylabels[values]: yaxis_values_per_bucket
    })

    fig = px.scatter(
        df,
        x=xlabels[values],
        y=ylabels[values],
        width=800,
        height=600,
        template="simple_white",
    ).update_traces(
        mode='markers+lines'
    ).update_layout(
        showlegend=False,
        font_size=30 if values.startswith("kl") else 25,
        title=dict(
            text=title,
            font=dict(
                size=20  # Smaller font size for title
            )
        )
    )

    xmin = df[xlabels[values]].min()
    xmax = df[xlabels[values]].max()
    fig.add_shape(type="line", x0=xmin, y0=xmin, x1=xmax, y1=xmax, line=dict(color="red", dash="dash", width=2), opacity=0.8)
    fig.add_hline(y=0, line=dict(color="red", dash="dash", width=2), opacity=0.8)

    if values.startswith("loss"):
        fig.add_vline(x=0, line_width=1, opacity=0.4, line_color="black")
        x_y_text_list = [(0.98, 0.54, "No intervention"), (0.98, 1.04, "Full ablation")]
    elif values.startswith("kl-div"):
        x_y_text_list = [(0.98, 0.17, "No intervention"), (0.98, 0.86, "Full ablation")]

    for (x, y, text) in x_y_text_list:
        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=x, y=y,
            font_size=14,
            xanchor="right",
            showarrow=False,
            align="left",
        )

    fig.update_xaxes(showspikes=True, spikemode="across")
    fig.update_yaxes(showspikes=True, spikethickness=2)
    fig.update_layout(spikedistance=1000, hoverdistance=100)

    fig.show()
    return fig


def generate_scatter(
    cspa_results: Dict[str, Tensor], 
    DATA_STR_TOKS_PARSED: List[List[str]],
    subtext_to_cspa: List[str] = ["i.e. ablate everything", "except the pure copy-", "suppression mechanism"],
    cspa_y_axis_title: str = "CSPA",
    show_key_results: bool = False,
    batch_index_colors_to_highlight: List[int] = [],
):
    # Get results
    l_orig = cspa_results["loss"]
    l_cspa = cspa_results["loss_cspa"]
    l_abl = cspa_results["loss_ablated"]
    # Get the 2.5% cutoff examples, and the 5% cutoff examples

    A_m_O = l_abl - l_orig
    C_m_O = l_cspa - l_orig

    x = A_m_O.flatten()
    y = C_m_O.flatten()
    # colors = y.abs() / (x.abs() + y.abs())
    # colors_reshaped = colors.reshape(l_orig.shape)
    colors = {i: [] for i in batch_index_colors_to_highlight}

    text = []
    batch_size, seq_len_minus_1 = l_orig.shape
    for bi in range(batch_size):
        for si in range(seq_len_minus_1):
            text.append("<br>".join([
                f"Sequence {bi}, token {si}",
                f"<br>Orig loss = {l_orig[bi, si]:.3f}",
                f"Ablated loss = {l_abl[bi, si]:.3f}",
                f"CS loss = {l_cspa[bi, si]:.3f}<br>", 
                # f"<br>y/(y+x) = {colors_reshaped[bi, si]:.1%}<br>",
                "".join(DATA_STR_TOKS_PARSED[bi][max(0, si-10): si+1]) + f" ➔ <b>{DATA_STR_TOKS_PARSED[bi][si+1]}</b>"
            ]))
            for k in colors:
                colors[k].append(bi < k)
            
    fig_dict = {}

    # Get a dataframe with one color list for each of the different batch index cutoff points we have
    df = pd.DataFrame({
        "x": utils.to_numpy(x), "y": utils.to_numpy(y), "text": text, 
        **{f"colors_{batch_idx}": utils.to_numpy(color_list) for batch_idx, color_list in colors.items()}
    })

    # For each batch index cutoff point, we get a new figure (because that's how the data gets split)
    for batch_idx in batch_index_colors_to_highlight:

        fig = px.scatter(
            df, x="x", y="y", custom_data="text", width=1000, height=700, 
            color=f"colors_{batch_idx}", color_discrete_sequence=["red", "rgba(20, 20, 220, 0.2)"],
        ).update_layout(
            template="simple_white",
            title="Comparison of change in loss for different kinds of ablation",
            yaxis=dict(tickmode="linear", dtick=0.5, title_text="", scaleanchor="x", scaleratio=1, title_font_size=18),
            xaxis=dict(tickmode="linear", dtick=0.5, title_text="Full ablation", title_font_size=18),
            showlegend=False,
            title_font_size=20,
            width = 1200, 
            height = 800,
            margin = dict(b=100, r=200, l=220, t=120),
        ).update_traces(
            hovertemplate="%{customdata}",
        )

        AO_mean = A_m_O.mean().item()
        AO_std = A_m_O.std().item()
        CO_mean = C_m_O.mean().item()
        CO_std = C_m_O.std().item()

        fig.add_hline(y=CO_mean, line_width=2, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=CO_mean+CO_std, line_width=2, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=CO_mean-CO_std, line_width=2, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_vline(x=AO_mean, line_width=2, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_vline(x=AO_mean+AO_std, line_width=2, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_vline(x=AO_mean-AO_std, line_width=2, line_dash="dash", line_color="red", opacity=0.5)

        for text, xref, yref, x, y, xanchor in zip(
            [f"Mean = {CO_mean:.4f}", f"Std = {CO_std:.4f}", f"Mean = {AO_mean:.4f}", f"Std = {AO_std:.4f}"],
            ["paper", "paper", "x", "x"],
            ["y", "y", "paper", "paper"],
            [1.0, 0.88, AO_mean, AO_mean-AO_std],
            [CO_mean+0.1*CO_std, CO_mean-2.5*CO_std, 1.05, 1.00],
            ["left", "left", "center", "right"]
        ):
            fig.add_annotation(
                text=f" {text} ",
                xref=xref, yref=yref,
                x=x, y=y,
                font_size=16,
                xanchor=xanchor,
                showarrow=False,
                align=xanchor,
            )

        fig.add_annotation(
            text=cspa_y_axis_title,
            # text="<br>".join(["CS-preserving", "ablation"]),
            xref="paper", yref="paper",
            x=-0.05, y=0.55, # 0.57
            font_size=18,
            xanchor="right",
            showarrow=False,
            align="right",
        )
        fig.add_annotation(
            text="<br>".join(subtext_to_cspa),
            xref="paper", yref="paper",
            x=-0.05, y=0.48,
            font_size=12,
            xanchor="right",
            showarrow=False,
            align="right",
        )

        # if show_key_results:
        #     fig.add_annotation(
        #         text="<br>".join([
        #             "Key results:",
        #             "",
        #             "1. CS-preserving ablation explains most of head's effect on loss<br>    (std dev is much smaller on y-axis)",
        #             "",
        #             "2. CS-preserving ablation performs as well as original model<br>    (mean is nearly zero on y-axis)",
        #         ]),
        #         xref="paper", yref="paper",
        #         x=0.55, y=1.00,
        #         showarrow=False,
        #         font_size=13,
        #         xanchor="left",
        #         align="left",
        #         bordercolor="black",
        #         borderpad=10,
        #     )
        fig.data[0].marker.size = 4
        if len(fig.data) > 1:
            fig.data[1].marker.size = 2
        fig_dict[batch_idx] = fig

    fig_samecolor = copy(fig)
    fig_samecolor.data[0].marker.color = "rgba(20, 20, 220, 0.5)"
    fig_samecolor.data[0].marker.size = 3
    if len(fig.data) > 1:
        fig_samecolor.data[1].marker.color = "rgba(20, 20, 220, 0.5)"
        fig_samecolor.data[1].marker.size = 3
    fig_samecolor.show()

    fig_dict["samecolor"] = fig_samecolor
    return fig_dict









def add_cspa_to_streamlit_page(
    cspa_results: Dict[str, Float[Tensor, "batch seq"]],
    s_sstar_pairs: Dict[Tuple[int, int], List[Tuple[int, int]]],
    data_str_toks_parsed: List[List[str]],
    model: HookedTransformer,
    html_plots_filename: Optional[str] = None,
    HTML_PLOTS: Optional[dict] = None,
    toks_for_doing_DLA: Optional[Int[Tensor, "batch seq"]] = None,
    test_idx: Optional[int] = None,
    cutoff: float = 0.03,
    verbose: bool = False,
): 
    '''
    Adds CSPA results as a fifth tab to the Streamlit page.

    Also has the option to add logits and DLA below the logits & DLA plot on the second tab (so you can
    do things like compare DLA in CSPA to regular DLA, and see where CSPA's DLA is falling short of the
    head's actual DLA.

    You have to specify either html_plots_filename or give HTML_PLOTS, but not both.

    Also, if cspa_results and s_sstar_pairs are actually {str: {str: Tensor}} rather than {str: Tensor},
    this is interpreted as being multiple different visualisations which you can compare. They're stacked
    on top of each other in the HTML page.
    '''
    assert (html_plots_filename is None) != (HTML_PLOTS is None)

    cspa_values = list(cspa_results.values())
    if isinstance(cspa_values[0], t.Tensor):
        cspa_results = {"": cspa_results}
        s_sstar_pairs = {"": s_sstar_pairs}
    first_key = list(cspa_results.keys())[0]

    # ! Get all CSPA visulisations (this looks janky because I like being able to compare different forms of CSPA!)
    if verbose: print("Generating CSPA plots  ...", end="\r"); t0 = time.time()
    # First, get them in the form {CSPA type: {batch_idx: single plot}}
    CSPA_PLOTS = {
        k: generate_html_for_cspa_plots(
            str_toks_list = data_str_toks_parsed,
            cspa_results = cspa_results[k],
            s_sstar_pairs = s_sstar_pairs[k],
            cutoff = cutoff,
        )
        for k in cspa_results
    }
    # Second, get them in the form {batch_idx: (multiples plots, concatenated over type)}}
    CSPA_PLOTS = {
        (batch_idx,): "".join([f"<h3>{k}</h3>" + CSPA_PLOTS[k][(batch_idx,)] for k in CSPA_PLOTS])
        for batch_idx in range(cspa_results[first_key]["loss"].shape[0])
    }
    if verbose: print(f"Generating CSPA plots  ... {time.time()-t0:.2f}")

    # Display a single plot and stop here, if requested
    if test_idx is not None:
        display(HTML(CSS.replace("min-width: 275px", "min-width: 200px") + CSPA_PLOTS[(test_idx,)] + "<br>" * 20))
        return CSPA_PLOTS

    # Open the HTML plots
    if HTML_PLOTS is None:
        full_filename: Path = ST_HTML_PATH / html_plots_filename
        with gzip.open(full_filename, "rb") as f:
            HTML_PLOTS = pickle.load(f)
    
    # Check that the CSPA results we have are as large as these html plots
    # If CSPA plots are smaller, we only show as many as are already present in the HTML plots dict
    html_plots_batch_idx = max(keys[0] for keys in HTML_PLOTS["DLA"]) + 1
    cspa_batch_idx, seq_len_minus1 = cspa_results[first_key]["loss"].shape
    assert cspa_batch_idx >= html_plots_batch_idx
    CSPA_PLOTS = {k: v for k, v in CSPA_PLOTS.items() if k[0] < html_plots_batch_idx}

    # Set the CSPA results
    HTML_PLOTS["CSPA"] = CSPA_PLOTS

    # If we're adding DLA & logits, then generate the plots for this and add them to the big dict
    if toks_for_doing_DLA is not None:

        if verbose: print("Generating DLA plots ...", end="\r"); t0 = time.time()
        toks_cpu = toks_for_doing_DLA[:html_plots_batch_idx].cpu()

        # ! Get all DLA visulisations (this looks janky because I like being able to compare different forms of CSPA!)
        html_pos_and_neg = {
            k: generate_html_for_DLA_plot(
                toks = toks_cpu,
                dla_logits = cspa_results[k]["dla"][:html_plots_batch_idx],
                model = model
            )
            for k in cspa_results
        }
        html_pos_and_neg = {
            batch_idx: [
                "".join([f"<h3>{k}</h3>" + html_pos_and_neg[k][0][batch_idx] for k in html_pos_and_neg]), # pos
                "".join([f"<h3>{k}</h3>" + html_pos_and_neg[k][1][batch_idx] for k in html_pos_and_neg]), # neg
            ]
            for batch_idx in range(html_plots_batch_idx)
        }

        for batch_idx, (html_pos, html_neg) in html_pos_and_neg.items():
            HTML_PLOTS["DLA"][(batch_idx, "CSPA", "neg")] = str(html_neg)
            HTML_PLOTS["DLA"][(batch_idx, "CSPA", "pos")] = str(html_pos)

        if verbose: print(f"{'Generating DLA plots':<22} ... {time.time()-t0:.2f}")
        if verbose: print(f"{'Generating logit plots':<22} ...", end="\r"); t0 = time.time()

        # ! Get all logit visulisations (this can look janky because I like being able to compare different forms of CSPA!)
        if not isinstance(cspa_results, dict): cspa_results = {"": cspa_results}
        html_logits_list = {
            k: generate_html_for_logit_plot(
                full_toks = toks_cpu,
                full_logprobs = cspa_results[k]["logits_cspa"][:html_plots_batch_idx].log_softmax(-1),
                full_non_ablated_logprobs = cspa_results[k]["logits_orig"][:html_plots_batch_idx].log_softmax(-1),
                model = model,
            )
            for k in cspa_results
        }
        html_logits_list = {
            batch_idx: "".join([f"<h3>{k}</h3>" + html_logits_list[k][batch_idx] for k in html_logits_list])
            for batch_idx in range(html_plots_batch_idx)
        }
        for batch_idx, html in html_logits_list.items():
            HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, "CSPA")] = str(html)
        
        if verbose: print(f"{'Generating logit plots':<22} ... {time.time()-t0:.2f}")

    # Save new plots (or just return)
    if html_plots_filename is None:
        return HTML_PLOTS
    else:
        if verbose: print(f"Saving plots to {full_filename.name!r} ...", end="\r"); t0 = time.time()
        with gzip.open(full_filename, "wb") as f:
            pickle.dump(HTML_PLOTS, f)
        if verbose: print(f"Saving plots to {full_filename.name!r} ... {time.time()-t0:.2f}", end="\n")