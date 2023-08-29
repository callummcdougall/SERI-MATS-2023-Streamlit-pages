import sys, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
st.set_page_config(layout="wide")

import plotly.express as px
from pathlib import Path

for st_page_dir in [
    os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0] + "SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("seri-mats-2023-streamlit-pages")[0] + "seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("/app/seri-mats-2023-streamlit-pages")[0] + "/app/seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
    "/mount/src/seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
]:
    if os.path.exists(st_page_dir): break
else: raise Exception("Couldn't find root dir")
root_dir = st_page_dir.replace("/transformer_lens/rs/callum2/st_page", "")
ST_HTML_PATH = Path(st_page_dir) / "media"
if sys.path[0] != root_dir: sys.path.insert(0, root_dir)

from transformer_lens.rs.callum2.st_page.streamlit_styling import styling
from transformer_lens.rs.callum2.utils import ST_HTML_PATH
styling()

import pandas as pd
import numpy as np
import pickle
from typing import Literal

import torch as t
t.set_grad_enabled(False)

# st.set_page_config(layout="wide")

param_sizes = {
    "gpt2-small": "85M",
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

# @st.cache_data(show_spinner=False, max_entries=1)
def load_results():
    results = pickle.load(open(ST_HTML_PATH / "anti_induction/scores_dict.pkl", "rb"))
    results = {k: v for k, v in results.items()}
    return results

# ! TODO - figure out why OPT is weird, no copy suppression anywhere
SCORES_DICT = load_results()
MODEL_NAMES = sorted(SCORES_DICT.keys())

min_size = int(min([get_size(model_name) for model_name in MODEL_NAMES]) // 1e6)
max_size = int(max([get_size(model_name) for model_name in MODEL_NAMES]) // 1e6)

def plot_all_results(
    pospos=False,
    showtext=False,
    fraction=False,
    categories="all",
    cs_metric:Literal["ioi", "norm"]="ioi",
    size_range=(min_size, max_size)
):
    results_copy_suppression_ioi = []
    results_anti_induction = []
    results_copy_suppression_norm = []
    model_names = []
    head_names = []
    fraction_list = []
    num_params = []

    for model_name in MODEL_NAMES:
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

    df = pd.DataFrame({
        "results_cs_ioi": results_copy_suppression_ioi,
        "results_cs_norm": results_copy_suppression_norm,
        "results_ai_rand": results_anti_induction,
        "model_name": model_names,
        "Head name": head_names,
        "head_and_model_names": [f"{model_names[i]} [{head_names[i]}]" for i in range(len(model_names))],
        "fraction_list": fraction_list,
        "num_params": num_params,
    })

    if cs_metric == "norm":
        df = df[df["results_cs_norm"] != np.nan]
        x = "results_cs_norm"
    else:
        x = "results_cs_ioi"

    if pospos:
        is_pos = [i for i, (x, y) in enumerate(zip(results_copy_suppression_ioi, results_anti_induction)) if x > 0 and y > 0]
        df = df.iloc[is_pos]

    if categories.lower() == "none":
        df = df[df["model_name"] == ""] # filter everything out
    elif categories.lower() != "all":
        df = df[[categories.lower() in name for name in df["model_name"]]]

    df = df[df["num_params"] >= size_range[0] * 1e6]
    df = df[df["num_params"] <= size_range[1] * 1e6]

    fig = px.scatter(
        df,
        x=x, 
        y="results_ai_rand", 
        color='model_name' if not(fraction) else "fraction_list", 
        hover_name="head_and_model_names",
        hover_data={
            "Copy Suppression": [f"<b>{x:.3f}</b>" for x in df[x]],
            "Anti-Induction": [f"<b>{x:.3f}</b>" for x in df["results_ai_rand"]],
            "model_name": False,
            x: False,
            "results_ai_rand": False,
        },
        text="head_and_model_names" if showtext else None,
        title="Anti-Induction Scores (repeated random tokens) vs Copy-Suppression Scores (IOI)",
        labels={
            x: "Copy-Suppression Score",
            "results_ai_rand": "Anti-Induction Score",
            "model_name": ""
        },
        height=550,
        color_continuous_scale=px.colors.sequential.Rainbow if fraction else None,
    )
    # fig.update_layout(legend_title_font_size=18)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_traces(textposition='top center')
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.add_hline(y=0, line_width=1, line_color="black")

    # Now we add legend groups
    for trace in fig.data:
        for group_substr in ["GPT", "OPT", "Pythia", "SoLU", "GELU", "Other"]:
            if group_substr.lower() in trace.name.lower():
                break
        trace.legendgroup = group_substr
        trace.legendgrouptitle = {'text': f"<br><span style='font-size:16px'>{'='*8} {group_substr} {'='*8}</span>"}

    # fig.update_layout(paper_bgcolor='rgba(255,244,214,0.5)', plot_bgcolor='rgba(255,244,214,0.5)')
    return fig


st.markdown(
r"""
# Anti-Induction vs Copy-Suppression

The plot below shows the scores of various heads for anti-induction and copy-suppression.

An explanation of what the metrics are:

### Anti-induction

Given a random repeated sequence `AB ... AB ...` of length $2 \times 30$ (with BOS prepended), what is the average direct logit effect of this attention head on the correct token in the second half of the sequence, only coming from the source token which is one position after the first instance of the destination token? (i.e. the classic induction algorithm)

### Copy suppression

Given a sequence such as ***"When John and Mary went to the shops, John gave a drink to Mary"***, what is the average direct effect on the logits for the `IO` token (`" Mary"`), resulting just from the head's attention from `end` (`" to"`) to `IO` (`" Mary"`)?

For each attention head, we used 100 randomly generated sequences.
""")

st.info(r"""Coming soon - copy-suppression measured by performance recovery on CSPA, rather than by behaviour on IOI.""")

cols = st.columns(2)
with cols[0]:
    pospos = st.checkbox("Only show pos-pos quadrant", True)
    showtext = st.checkbox("Show head names as annotations")
    fraction = st.checkbox("Color points by fraction through model")
    categories = st.radio(
        "Filter by model class (you can also click on the legend to filter):", 
        ["all", "none"] + sorted(["GPT", "OPT", "Pythia", "SoLU", "GeLU"])
    )
    # cs_metric = st.radio(
    #     "Copy-suppression metric (suppression of IOI, or projection onto top unembedding)",
    #     ["ioi", "norm"]
    # )
    cs_metric = "ioi"
    size_range = st.slider(
        label="Filter by number of params in model",
        min_value=min_size,
        max_value=max_size, 
        value=(min_size, max_size), 
        step=1,
        format='%2fM'
    )

fig = plot_all_results(
    pospos=pospos,
    showtext=showtext,
    fraction=fraction,
    categories=categories,
    cs_metric=cs_metric,
    size_range=size_range
)

st.plotly_chart(fig, use_container_width=True)