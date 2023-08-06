import sys, os
for root_dir in [
    os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts", # For Arthur's branch
    "/app/seri-mats-2023-streamlit-pages/explore_prompts", # For Streamlit page (public)
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/explore_prompts", # For Arthur's branch
    os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0] + "SERI-MATS-2023-Streamlit-pages/explore_prompts", # For Arthur's branch
]:
    if os.path.exists(root_dir):
        break
os.chdir(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

from typing import List, Tuple, Literal, Union, Dict, Optional
import torch as t
from torch import Tensor
from jaxtyping import Float
import plotly.express as px
from copy import copy
from transformer_lens import utils, ActivationCache, HookedTransformer
import pandas as pd
from explore_prompts_utils import create_title_and_subtitles

def generate_scatter(
    ICS: Dict[str, Tensor], # the thing we get from MODEL_RESULTS.is_copy_suppression
    DATA_STR_TOKS_PARSED: List[List[str]],
    subtext_to_cspa: List[str] = ["i.e. ablate everything", "except the pure copy-", "suppression mechanism"],
    cspa_y_axis_title: str = "CSPA",
    show_key_results: bool = True,
    title: str = "Comparison of change in loss for different kinds of ablation",
):
    # Get results
    l_orig = ICS["L_ORIG"]
    l_cspa = ICS["L_CS"]
    l_abl = ICS["L_ABL"]
    # Get the 2.5% cutoff examples, and the 5% cutoff examples

    A_m_O = l_abl - l_orig
    C_m_O = l_cspa - l_orig

    x = A_m_O.flatten()
    y = C_m_O.flatten()
    # colors = y.abs() / (x.abs() + y.abs())
    # colors_reshaped = colors.reshape(l_orig.shape)
    colors = []

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
            colors.append(bi==24)



    df = pd.DataFrame({
        "x": utils.to_numpy(x), "y": utils.to_numpy(y), "colors": utils.to_numpy(colors), "text": text, 
    })

    fig = px.scatter(
        df, x="x", y="y", custom_data="text", opacity=0.5, width=1000, height=700, # color="colors"
        # marginal_x="histogram", marginal_y="histogram",
        # trendline="ols",
        # color="colors", color_continuous_scale=colorscheme, range_color=(0, 1),
    ).update_layout(
        template="simple_white",
        title=title,
        yaxis=dict(tickmode="linear", dtick=0.5, title_text=""),
        xaxis=dict(tickmode="linear", dtick=0.5, title_text="Full ablation"),
    ).update_traces(
        hovertemplate="%{customdata}",
        marker_size=3,
        # selector=dict(type="scatter"),
    )
    results = px.get_trendline_results(fig)

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
            text=" " + text + " ",
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

    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        margin=dict.fromkeys("tblr", 100),
        # title_y=0.94,
        width = 1200,
        height = 800,
        margin_r = 200,
        margin_l = 220,
        margin_t = 120,
        yaxis = dict(scaleanchor="x", scaleratio=1),
    )
    fig2 = copy(fig)
    if show_key_results:
        fig2.add_annotation(
            text="<br>".join([
                "Key results:",
                "",
                "1. CS-preserving ablation explains most of head's effect on loss<br>    (std dev is much smaller on y-axis)",
                "",
                "2. CS-preserving ablation performs as well as original model<br>    (mean is nearly zero on y-axis)",
            ]),
            xref="paper", yref="paper",
            x=0.55, y=1.00,
            showarrow=False,
            font_size=13,
            xanchor="left",
            align="left",
            bordercolor="black",
            borderpad=10,
        )
    fig2.show()
    return fig, results, df



def generate_hist(ICS: Dict[str, Tensor], threshold: float):

    l_orig = ICS["L_ORIG"]
    l_cspa = ICS["L_CS"]
    l_abl = ICS["L_ABL"]

    l_cspa_minus_orig = l_cspa - l_orig
    l_abl_minus_orig = l_abl - l_orig

    is_top_5pct_ablated_loss = (l_abl_minus_orig > l_abl_minus_orig.quantile(1-threshold)) | (l_abl_minus_orig < l_abl_minus_orig.quantile(threshold))

    loss_ratio_for_top_5pct = (l_cspa_minus_orig / l_abl_minus_orig).flatten()[is_top_5pct_ablated_loss.flatten()]

    fig = px.histogram(
        utils.to_numpy(loss_ratio_for_top_5pct),
        template="simple_white",
        title=create_title_and_subtitles(
            f"Effect of CSPA relative to full ablation (top {threshold:.1%})",
            ["If this is 0, then the copy-suppression mechanism fully explains the head.", "If this is 1, then the copy-suppression mechanism explains nothing."]
        ),
        width=800,
        labels={"x": "Loss ratio: (CSPA - ORIG) / (ABLATED - ORIG)"},
    ).update_layout(
        hovermode="x unified",
        showlegend=False,
        title_y=0.9,
        margin_t=150,
    )
    median = loss_ratio_for_top_5pct.median().item()
    fig.add_vline(
        x=median, line_width=3, line_dash="dash", line_color="red", opacity=1.0,
        annotation_text=f" Median = {median:.3f}"
    )
    fig.show()
    return fig





# def func(
#     MODEL_RESULTS: ModelResults,
#     effect: Literal["direct", "indirect", "both"] = "direct",
#     ln_mode: Literal["frozen", "unfrozen"] = "frozen",
#     ablation_mode: Literal["mean", "zero"] = "mean",
#     head: Tuple[int, int] = (10, 7)
# ):
#     layer, head = head

#     # ! Do heatmaps of loss diffs

#     # Get results
#     ICS = MODEL_RESULTS.is_copy_suppression[(effect, ln_mode, ablation_mode)][10, 7]
#     z = ICS["CS"]
#     ratio = ICS["LR"]
#     l_orig = ICS["L_ORIG"]
#     l_cs = ICS["L_CS"]
#     l_abl = ICS["L_ABL"]
#     # Get the 2.5% cutoff examples, and the 5% cutoff examples
#     l_abl_minus_orig = l_abl - l_orig
#     non_extreme_05pct = (l_abl_minus_orig < l_abl_minus_orig.quantile(0.95)) & (l_abl_minus_orig > l_abl_minus_orig.quantile(0.05))
#     non_extreme_025pct = (l_abl_minus_orig < l_abl_minus_orig.quantile(0.975)) & (l_abl_minus_orig > l_abl_minus_orig.quantile(0.025))
#     z_05pct = t.where(non_extreme_05pct, 2, z)
#     z_025pct = t.where(non_extreme_025pct, 2, z)

#     fig = imshow(
#         t.stack([
#             l_cs - l_orig,
#             l_orig - l_abl,
#             l_cs - l_abl
#         ]),
#         facet_col = 0,
#         facet_labels = [
#             "CS minus orig<br>(this should be near zero when CS is actually happening, or when head just doesn't<br>affect loss in the first place)",
#             "Orig minus ablated<br>(reference point, to see the big examples)",
#             "CS minus ablated<br>(as a test, should be zero if CS is replaced w/ ablation)"
#         ],
#         title="Losses compared: original, ablation, CS-preserving ablation",
#         aspect="auto",
#         return_fig=True,
#         width=1800,
#         height=800,
#         border=True,
#     )
#     batch_size, seq_len = DATA_TOKS.shape
#     text = []
#     for bi in range(batch_size):
#         text.append([])
#         for si in range(seq_len-1):
#             text[-1].append("<br>".join([
#                 f"<br>Orig loss = {l_orig[bi, si]:.2f}",
#                 f"Ablated loss = {l_abl[bi, si]:.2f}",
#                 f"CS loss = {l_cs[bi, si]:.2f}", 
#                 f"<br>Loss ratio = {ratio[bi, si]:.2f}<br>",
#                 "".join(DATA_STR_TOKS_PARSED[bi][max(0, si-10): si+1]) + f" ➔ <b>{DATA_STR_TOKS_PARSED[bi][si+1]}</b>"
#             ]))
#     for i in range(3):
#         fig.data[i]["customdata"] = text
#         fig.data[i]["hovertemplate"] = "Sequence: %{x} <br>Token index: %{y} <br>%{customdata}"
#     fig.update_layout(
#         autosize=False,
#         margin_t=150,
#         title_y = 0.94,
#     )
#     fig.show()

#     # ! Do heatmap of classifications (red / green / grey)

#     fig = imshow(
#         t.stack([z_05pct, z_025pct]),
#         facet_col=0,
#         facet_labels=["75% ratio threshold,<br>only showing top 5% of loss examples", "75% ratio threshold,<br>only showing top 2.5% of loss examples"],
#         title = "Copy suppression classifications",
#         color_continuous_scale = [
#             (0.0, "red"), (0.33, "red"),
#             (0.33, "#2CA02C"), (0.66, "#2CA02C"),
#             (0.66, "#eee"), (1.0, "#eee"),
#         ],
#         return_fig = True,
#         range_color = [-0.5, 2.5],
#         aspect = "auto",
#         border = True,
#     )
#     for i in range(2):
#         fig.data[i]["customdata"] = text
#         fig.data[i]["hovertemplate"] = "Sequence: %{x} <br>Token index: %{y} <br>%{customdata}"

#     fig.update_layout(
#         coloraxis_colorbar=dict(
#             title = "Classification",
#             tickvals = [0, 1, 2],
#             ticktext = ["Not copy-suppression", "Copy-suppression", "Inconclusive"],
#             lenmode = "pixels", 
#             len = 200,
#         ),
#         width = 1600,
#         height = 800,
#         autosize = False,
#     )
#     fig.show()

#     # ! Do scatterplot of examples

#     direct = MODEL_RESULTS.loss_diffs[("direct", ln_mode, ablation_mode)][layer, head]
#     indirect = MODEL_RESULTS.loss_diffs[("indirect", ln_mode, ablation_mode)][layer, head]
#     direct_mean = direct.mean().item()
#     indirect_mean = indirect.mean().item()

#     for s, v, z in zip(["5%", "2.5%"], [0.05, 0.025], [z_05pct, z_025pct]):

#         direct_top_quantile = direct.quantile(1-v).item()
#         direct_bottom_quantile = direct.quantile(v).item()
#         colors = ["red", "#2CA02C", "#ccc"]
#         colors = [colors[i.item()] for i in z.flatten()]
#         size = [7 if i.item() == 2 else 7 for i in z.flatten()]

#         fig = go.Figure(
#             layout = go.Layout(
#                 template="simple_white",
#                 width=1000,
#                 height=700,
#                 title="<br>".join([
#                     f"Direct vs. Indirect effect for head {layer}.{head}",
#                     f"<span style='font-size:13px'>Ablation type: {ablation_mode}-ablation, layernorm {ln_mode}, {effect} effect measured</span>",
#                     f"<span style='font-size:13px'>Loss ratio threshold = 0.75<br>Filtered by top/bottom 5% of loss-affecting examples",
#                 ]),
#                 xaxis_title_text="Direct effect (positive ⇒ head is helpful)",
#                 yaxis_title_text="Indirect effect (positive ⇒ head is helpful)",
#             )
#         )
#         fig.add_trace(go.Scattergl(
#             x = utils.to_numpy(direct.flatten()),
#             y = utils.to_numpy(indirect.flatten()),
#             mode = "markers",
#             marker = dict(
#                 opacity = 0.7,
#                 size = size,
#                 color = colors,
#             ),
#             hovertext = [te for tex in text for te in tex], # flatten hackily
#             showlegend=False,
#         ))
#         fig.add_hline(y=indirect_mean, line_width=3, line_dash="dash", line_color="red")
#         fig.add_vline(x=direct_mean, line_width=3, line_dash="dash", line_color="red")
#         fig.add_vline(x=direct_top_quantile, line_width=3, line_dash="dash", line_color="red")
#         fig.add_vline(x=direct_bottom_quantile, line_width=3, line_dash="dash", line_color="red")
#         fig.add_trace(go.Scatter(
#             x=[1.5],
#             y=[indirect_mean+0.01],
#             mode="text",
#             text=[f"Mean = {indirect_mean:.3f}"],
#             textfont=dict(size=15),
#             textposition="top center",
#             showlegend=False,
#         ))
#         y_max = indirect.max().item()
#         y_min = indirect.min().item()
#         fig.add_trace(go.Scatter(
#             x=[direct_bottom_quantile-0.02],
#             y=[y_min + 0.1 * (y_max - y_min)],
#             mode="text",
#             text=[f"Bottom 5% = {direct_bottom_quantile:.3f}"],
#             textfont=dict(size=15),
#             showlegend=False,
#             textposition="middle left"
#         ))
#         fig.add_trace(go.Scatter(
#             x=[direct_mean+0.02, direct_top_quantile+0.02],
#             y=[y_max + 0.1 * (y_min - y_max), y_min + 0.1 * (y_max - y_min)],
#             mode="text",
#             text=[f"Mean = {direct_mean:.3f}", f"Top 5% = {direct_top_quantile:.3f}"],
#             textfont=dict(size=15),
#             showlegend=False,
#             textposition="middle right"
#         ))

#         fig.update_layout(
#             xaxis_title_font_size=18,
#             yaxis_title_font_size=18,
#             margin=dict.fromkeys("tblr", 100),
#             coloraxis_colorbar=dict(
#                 title = "Classification",
#                 tickvals = [0,1,2],
#                 ticktext = ["Not copy-suppression", "Copy-suppression", "Inconclusive"],
#                 lenmode = "pixels", 
#                 len = 200,
#             ),
#             title = dict(
#                 y = 0.94,
#             ),
#             width = 1200,
#             height = 800,
#             margin_r = 200,
#             margin_t = 150,
#         )
#         num_neg = (z==0).int().sum()
#         num_pos = (z==1).int().sum()

#         display(HTML(f"""
#     <h2>Classification Results</h2><br><div style='font-size:16px;'>
#     These are the classification results, with a loss threshold of 75% (i.e. loss ratio must be above 0.75 for a positive classification).

#     Of the top and bottom {s} of results, the breakdown was as follows:<br><ol>
#     <li><b>{num_pos}/{num_neg+num_pos} = {num_pos/(num_neg+num_pos):.2%}</b> were classified as <b>copy-suppression</b></li>
#     <li><b>{num_neg}/{num_neg+num_pos} = {num_neg/(num_neg+num_pos):.2%}</b> were classified as <b>not copy-suppression</b></li></ol>
#     </div>
#     """))
#         fig.add_annotation(
#             text="<br>".join([
#                 "<b><span style='color:#2CA02C'>Green</span></b> = Positive (this is copy-suppression)",
#                 "<b><span style='color:red'>Red</span></b>    = Negative (this is not copy-suppression)",
#                 "<b>Grey</b>   = Null classification (effect of ablation on<br>loss was small, so classification not attempted)",
#                 f"<br>{num_pos/(num_neg+num_pos):.2%} of classified points were positive."
#             ]),
#             xref="paper", yref="paper",
#             x=0.7, y=1.15,
#             showarrow=False,
#             font_size=14,
#             xanchor="left",
#             align="left",
#             bordercolor="black",
#             borderpad=10,
#         )
#         fig.show()





def plot_logit_lens(
    points_to_plot,
    resid_pre_head: ActivationCache,
    model: HookedTransformer,
    DATA_STR_TOKS_PARSED: List[List[str]],
    neg: bool = False,
    k: int = 15,
    title: Optional[str] = None,
):

    logits: Float[Tensor, "batch seq d_vocab"] = resid_pre_head @ model.W_U

    for seq, pos, expected in points_to_plot:
        if isinstance(expected, str):
            expected = [expected]
        s = f"{''.join(DATA_STR_TOKS_PARSED[seq][pos-4:pos+1])!r} -> {DATA_STR_TOKS_PARSED[seq][pos+1]!r} (expected {', '.join(list(map(repr, expected)))})"
        logits_topk = logits[seq, pos].topk(k, dim=-1, largest=not(neg))
        x = list(map(repr, model.to_str_tokens(logits_topk.indices)))
        y: list = logits_topk.values.tolist()
        color = ["#1F77B4"] * k

        # If the expected token is actually in the top k, then move it in there
        for str_tok_to_include in expected:
            tok_to_include = model.to_single_token(str_tok_to_include)
            for i, str_tok in enumerate(x):
                if repr(str_tok_to_include) == str_tok:
                    x[i] = x[i] + f" (#{i})"
                    color[i] = "#FF7F0E"
                    rank = i
                    break
            else:
                if neg: rank = (logits[seq, pos, tok_to_include] > logits[seq, pos]).sum().item()
                else: rank = (logits[seq, pos, tok_to_include] < logits[seq, pos]).sum().item()
                x.append(repr(str_tok_to_include)+ f" (#{rank})")
                y.append(logits[seq, pos, tok_to_include].item())
                color.append("#FF7F0E")

        # x = [f"{z} (#{i})" if not(z.endswith(")")) else z for i, z in enumerate(x)]

        px.bar(
            x=x, y=y, color=color, template="simple_white", title=f"({seq}, {pos}) {s}" if title is None else title,
            width=800, height=450, labels={"x": "Token", "y": "Logits", "color": "Token class"},
            color_discrete_map="identity"
        ).update_layout(
            xaxis_categoryorder = 'total ascending' if neg else 'total descending',
            hovermode="x unified", yaxis_range=[min(y) - 5, 0] if neg else [0, max(y) + 5], showlegend=False,
        ).show()



def plot_full_matrix_histogram(
    W_EE_dict: dict,
    src: Union[str, List[str]],
    dest: Union[str, List[str]],
    model: HookedTransformer,
    k: int = 10,
    head: Tuple[int, int] = (10, 7),
    neg: bool = True,
    circuit: Literal["QK", "OV"] = "OV",
    flip: bool = False,
):
    '''
    By default, this looks at what dest most attends to (QK) or what src causes to be most suppressed (OV).

    But if "flip" is True, then it looks at what things attend to src most (OV), or what causes dest to be most suppressed (OV).
    '''
    layer, head = head
    W_U: Tensor = W_EE_dict["W_U"].T
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    W_Q = model.W_Q[layer, head]
    W_K = model.W_K[layer, head]
    W_OV = W_V @ W_O
    W_QK = W_Q @ W_K.T
    denom = (model.cfg.d_head ** 0.5)
    b_Q: Tensor = model.b_Q[layer, head]
    b_K: Tensor = model.b_K[layer, head]

    if isinstance(src, str): src = [src]
    if isinstance(dest, str): dest = [dest]
    src_toks = list(map(model.to_single_token, src))
    dest_toks = list(map(model.to_single_token, dest))

    W_EE = W_EE_dict["W_E (including MLPs)"]

    if circuit == "OV":
        if flip:
            assert len(dest_toks) == 1
            hist_toks = src_toks
            W_U_toks = W_U.T[dest_toks[0]]
            W_EE_scaled = W_EE / W_EE.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_EE_scaled @ W_OV @ W_U_toks
        else:
            assert len(src_toks) == 1
            hist_toks = dest_toks
            W_EE_OV_toks = W_EE[src_toks[0]] @ W_OV
            W_EE_OV_scaled_toks = W_EE_OV_toks / W_EE_OV_toks.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_EE_OV_scaled_toks @ W_U

        full_vector_topk = full_vector.topk(k, dim=-1, largest=not(neg))

    elif circuit == "QK":
        if flip:
            assert len(src_toks) == 1
            hist_toks = dest_toks
            W_EE_scaled_toks = W_EE[src_toks[0]] / W_EE[src_toks[0]].std(dim=-1, keepdim=True)
            W_U_scaled = W_U.T / W_U.T.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_U_scaled @ (W_QK @ W_EE_scaled_toks + W_Q @ b_K) / denom
        else:
            assert len(dest_toks) == 1
            hist_toks = src_toks
            W_U_scaled_toks = W_U.T[dest_toks[0]] / W_U.T[dest_toks[0]].std(dim=-1, keepdim=True)
            W_EE_scaled = W_EE / W_EE.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = (W_U_scaled_toks @ W_QK + b_Q @ W_K.T) @ W_EE_scaled.T / denom

        full_vector_topk = full_vector.topk(k, dim=-1, largest=True)
    
    y = full_vector_topk.values.tolist()
    x = list(map(repr, model.to_str_tokens(full_vector_topk.indices)))
    color=["#1F77B4"] * k

    # If the expected token is actually in the top k, then move it in there
    for h_tok in hist_toks:
        h_str_tok = model.to_single_str_token(h_tok)
        for i, str_tok in enumerate(x):
            if str_tok == repr(h_str_tok):
                color[i] = "#FF7F0E"
                break
        else:
            if neg: rank = (full_vector[h_tok] > full_vector).sum().item()
            else: rank = (full_vector[h_tok] < full_vector).sum().item()
            x.append(repr(h_str_tok)+ f" (#{rank})")
            y.append(full_vector[h_tok].item())
            color.append("#FF7F0E")
    
    if circuit == "OV":
        if flip:
            title = f"<b style='font-size:22px;'>OV circuit</b>:<br>Which source tokens most suppress the prediction of<br><b>{dest[0]!r}</b> ?"
            x_label = "Source token"
        else:
            title = f"<b style='font-size:22px;'>OV circuit</b>:<br>Which predictions does source token <b>{src[0]!r}</b> suppress the most?"
            x_label = "Destination token (prediction)"
    else:
        if flip:
            title = f"<b style='font-size:22px;'>QK circuit</b>:<br>Which tokens' unembeddings most attend to source token <b>{src[0]!r}</b> ?"
            x_label = "Destination token (unembedding)"
        else:
            title = f"<b style='font-size:22px;'>QK circuit</b>:<br>Which source tokens does the unembedding of <b>{dest[0]!r}</b><br>attend to most?"
            x_label = "Source token"
    x_label = ""

    df = pd.DataFrame({
        "x": x, "y": y, "color": color
    })

    if neg:
        values_range=[min(y) - 10, 0]
        if max(y) > 0: values_range[1] = max(y) + 1
    else:
        values_range=[0, max(y) + 5]
        if min(y) < 0: values_range[0] = min(y) - 1

    fig = px.bar(
        df, x="y", y="x", color="color", template="simple_white", title=title,
        width=650, height=150+28*len(x), labels={"y": "Logits", "x": x_label, "color": "Token class"},
        color_discrete_map="identity", text_auto=".2f"
    ).update_layout(
        yaxis_categoryorder = 'total descending' if neg else 'total ascending',
        hovermode="y unified", xaxis_range=values_range, showlegend=False,
        margin_t=140, title_y=0.92,
    ).update_traces(
        textfont_size=13,
        textangle=0,
        textposition="outside",
        cliponaxis=False
    )
    fig.show()