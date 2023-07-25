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
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import gzip
import pickle
from pathlib import Path
from typing import List, Union, Literal
from torch import Tensor
from jaxtyping import Float
import plotly.express as px
import pandas as pd
import textwrap
import torch as t
t.set_grad_enabled(False)
# from transformers import AutoTokenizer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(layout="wide")

st.error("Note - I've not yet fixed scale factors in this page, so although the relative values should mostly make sense, the actual values might be incorrect.")

# dict_to_store = {
#     "tokenizer": model.tokenizer,
#     "W_V_107": model.W_V[10, 7],
#     "W_O_107": model.W_O[10, 7],
#     "W_V_1110": model.W_V[11, 10],
#     "W_O_1110": model.W_O[11, 10],
#     "W_Q_107": model.W_Q[10, 7],
#     "W_K_107": model.W_K[10, 7],
#     "W_Q_1110": model.W_Q[11, 10],
#     "W_K_1110": model.W_K[11, 10],
#     "b_Q_107": model.b_Q[10, 7],
#     "b_K_107": model.b_K[10, 7],
#     "b_Q_1110": model.b_Q[11, 10],
#     "b_K_1110": model.b_K[11, 10],
#     "W_EE": W_EE_dict["W_E (including MLPs)"],
#     "W_U": model.W_U,
# }
# dict_to_store_less = {
#     "tokenizer": model.tokenizer,
#     "W_EE_V": W_EE @ model.W_V[10, 7],
#     "W_U_O": model.W_O[10, 7] @ W_U,
#     "W_U_Q": W_U.T @ model.W_Q[10, 7],
#     "W_EE_K": W_EE @ model.W_K[10, 7],
#     "b_Q": model.b_Q[10, 7],
#     "b_K": model.b_K[10, 7],
# }

error_msg = """Token **`[{s}]`** (from {msg} token input) not in vocab!

Remember, by default you should append spaces at the start of the tokens you enter. You can use single or double quotes to specify your tokens, e.g. `' pier'` or `" pier"`.

You can enter multiple tokens in the second box by comma-separating them."""

short_error_msg = """

You should use single or double quotes to specify your tokens, e.g. `' pier'` or `" pier"`. When entering multiple tokens, they should be comma-separated. For example, the input `' pier', ' Pier'` will be accepted."""

def customwrap(s, width=70):
    return "<br>".join(textwrap.wrap(s, width=width))


def plot_full_matrix_histogram(
    dict_to_store_less: dict,
    src: Union[str, List[str]],
    dest: Union[str, List[str]],
    k: int = 10,
    head: str = "10.7",
    neg: bool = True,
    circuit: Literal["QK", "OV"] = "OV",
    flip: bool = False,
):
    '''
    By default, this looks at what dest most attends to (QK) or what src causes to be most suppressed (OV).

    But if "flip" is True, then it looks at what things attend to src most (OV), or what causes dest to be most suppressed (OV).
    '''
    _head = head.replace(".", "")
    tokenizer = dict_to_store_less["tokenizer"]
    W_EE_V = dict_to_store_less["W_EE_V"].float()
    W_U_O = dict_to_store_less["W_U_O"].float()
    W_U_Q = dict_to_store_less["W_U_Q"].float()
    W_EE_K = dict_to_store_less["W_EE_K"].float()
    # b_Q = dict_to_store_less["b_Q"].float()
    # b_K = dict_to_store_less["b_K"].float()
    # W_Q = dict_to_store["W_Q"].float()
    # W_K = dict_to_store["W_K"].float()
    # W_V = dict_to_store["W_V"].float()
    # W_O = dict_to_store["W_O"].float()

    # denom = (model.cfg.d_head ** 0.5)
    denom = 64 ** 0.5

    src_toks = tokenizer(src, return_tensors="pt")["input_ids"].squeeze().tolist()
    dest_toks = tokenizer(dest, return_tensors="pt")["input_ids"].squeeze().tolist()
    if isinstance(src, str): src = [src]
    if isinstance(src_toks, int): src_toks = [src_toks]
    if isinstance(dest, str): dest = [dest]
    if isinstance(dest_toks, int): dest_toks = [dest_toks]

    if circuit == "OV":
        if flip:
            if len(dest_toks) > 1:
                st.error(f"Error: if you're focusing on the destination token, you should only have a single destination token in the input. Instead you have {dest}.{short_error_msg}")
                return
            hist_toks = src_toks
            # W_U_toks = W_U.T[dest_toks[0]]
            # W_EE_scaled = W_EE / W_EE.std(dim=-1, keepdim=True)
            # W_EE_OV = W_EE_scaled @ W_OV
            # W_EE_OV_scaled = W_EE_OV / W_EE_OV.std(dim=-1, keepdim=True)
            # full_vector: Float[Tensor, "d_vocab"] = W_EE_OV_scaled @ W_U_toks
            W_U_O_toks_scaled = W_U_O.T[dest_toks[0]] / W_U_O.T[dest_toks[0]].std(dim=-1, keepdim=True)
            W_EE_V_scaled = W_EE_V / W_EE_V.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_EE_V_scaled @ W_U_O_toks_scaled
        else:
            if len(src_toks) > 1:
                st.error(f"Error: if you're focusing on the source token, you should only have a single source token in the input. Instead you have {src}.{short_error_msg}")
                return
            hist_toks = dest_toks
            # W_EE_scaled_toks = W_EE[src_toks[0]] / W_EE[src_toks[0]].std(dim=-1, keepdim=True)
            # W_EE_OV_toks = W_EE_scaled_toks @ W_OV
            # W_EE_OV_scaled_toks = W_EE_OV_toks / W_EE_OV_toks.std(dim=-1, keepdim=True)
            # full_vector: Float[Tensor, "d_vocab"] = W_EE_OV_scaled_toks @ W_U
            W_EE_V_toks_scaled = W_EE_V[src_toks[0]] / W_EE_V[src_toks[0]].std(dim=-1, keepdim=True)
            W_U_O_scaled = W_U_O / W_U_O.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_EE_V_toks_scaled @ W_U_O_scaled

        full_vector_topk = full_vector.topk(k, dim=-1, largest=not(neg))

    elif circuit == "QK":
        if flip:
            if len(src_toks) > 1:
                st.error(f"Error: if you're focusing on the source token, you should only have a single source token in the input. Instead you have {src}.{short_error_msg}")
                return
            hist_toks = dest_toks
            # W_EE_scaled_toks = W_EE[src_toks[0]] / W_EE[src_toks[0]].std(dim=-1, keepdim=True)
            # W_U_scaled = W_U.T / W_U.T.std(dim=-1, keepdim=True)
            # full_vector: Float[Tensor, "d_vocab"] = W_U_scaled @ (W_QK @ W_EE_scaled_toks + W_Q @ b_K) / denom
            W_EE_K_toks_scaled = W_EE_K[src_toks[0]] / W_EE_K[src_toks[0]].std(dim=-1, keepdim=True)
            W_U_Q_scaled = W_U_Q / W_U_Q.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_U_Q_scaled @ W_EE_K_toks_scaled / denom
        else:
            if len(dest_toks) > 1:
                st.error(f"Error: if you're focusing on the destination token, you should only have a single destination token in the input. Instead you have {dest}.{short_error_msg}")
                return
            hist_toks = src_toks
            # W_U_scaled_toks = W_U.T[dest_toks[0]] / W_U.T[dest_toks[0]].std(dim=-1, keepdim=True)
            # W_EE_scaled = W_EE / W_EE.std(dim=-1, keepdim=True)
            # full_vector: Float[Tensor, "d_vocab"] = (W_U_scaled_toks @ W_QK + b_Q @ W_K.T) @ W_EE_scaled.T / denom
            W_EE_scaled = W_EE_K / W_EE_K.std(dim=-1, keepdim=True)
            W_U_Q_toks_scaled = W_U_Q[dest_toks[0]] / W_U_Q[dest_toks[0]].std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_U_Q_toks_scaled @ W_EE_scaled.T / denom

        full_vector_topk = full_vector.topk(k, dim=-1, largest=True)
    
    y = full_vector_topk.values.tolist()
    x = list(map(lambda x: repr(x.replace("Ġ", " ")), tokenizer.batch_decode(full_vector_topk.indices)))
    color=["#1F77B4"] * k

    # If the expected token is actually in the top k, then move it in there
    for h_tok in hist_toks:
        h_str_tok = tokenizer.decode(h_tok)
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

            title = "<br><b style='font-size:22px;'>OV circuit</b>:<br>" + customwrap(f"Which source tokens most suppress the prediction of <b>{dest[0].replace(' ', 'Ġ')!r}</b> ?").replace("Ġ", " ")
            x_label = "Source token"
        else:
            title = "<br><b style='font-size:22px;'>OV circuit</b>:<br>" + customwrap(f"Which predictions does source token <b>{src[0].replace(' ', 'Ġ')!r}</b> suppress most, when attended to?").replace("Ġ", " ")
            x_label = "Destination token (prediction)"
    else:
        if flip:
            title = "<br><b style='font-size:22px;'>QK circuit</b>:<br>" + customwrap(f"Which tokens' unembeddings most attend to source token <b>{src[0].replace(' ', 'Ġ')!r}</b> ?").replace("Ġ", " ")
            x_label = "Destination token (unembedding)"
        else:
            title = "<br><b style='font-size:22px;'>QK circuit</b>:<br>" + customwrap(f"Which source tokens does the unembedding of <b>{dest[0].replace(' ', 'Ġ')!r}</b> attend to most?").replace("Ġ", " ")
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
        width=650, height=250+25*len(x), labels={"y": "Logits" if (circuit=="OV") else "Attention score", "x": x_label, "color": "Token class"},
        color_discrete_map="identity", text_auto=".2f"
    ).update_layout(
        yaxis_categoryorder = 'total descending' if neg else 'total ascending',
        hovermode="y unified", xaxis_range=values_range, showlegend=False,
        margin_t=120, margin_l=0, title_y=1, yaxis_tickfont_size=15, # + (0.92-0.98) * (len(x)/30)
    ).update_traces(
        textfont_size=14,
        textangle=0,
        textposition="outside",
        cliponaxis=False
    )
    return fig



ST_HTML_PATH = Path(root_dir) / "media"

@st.cache_data(show_spinner=False, max_entries=1)
def get_dict_to_store_less():
    return pickle.load(gzip.open(ST_HTML_PATH / f"OV_QK_circuits_less.pkl", "rb"))

dict_to_store_less = get_dict_to_store_less()

st.markdown(
r"""
# OV and QK circuits

Our theory for the mechanistic story behind pure copy-suppression can be summarized as follows:

* If we are predicting a token $S$ at destination position $D$ (in other words there is a large component of $\left(W_U\right)_{[:, S]}$ in the residual stream vector at this position), and $S$ also appeared earlier in context, then $D$ will attend back to $S$.
* The vector which gets moved from $S$ to $D$ is approximately the negative of the unembedding of $S$, in other words the prediction of $S$ is suppressed.

Mathematically, this means the following:

$$
\begin{aligned}
S &= \argmax_T \left\{ \left(W_{QK,\text{ full}}\right)_{T, S} \right\} \\
T &= \argmax_S \left\{ \left(W_{OV,\text{ full}}\right)_{S, T} \right\}
\end{aligned}
$$

Where these "full circuit" matrices are defined as follows:

$$
\begin{alignat*}{3}
W_{QK} \;&=\; W_Q W_K^T \; &&\in \mathbb{R}^{d_{model} \times d_{model}} \\
W_{QK,\text{ full}} \;&=\; W_{U}^T W_{QK} W_{EE}^T \; &&\in \mathbb{R}^{d_{model} \times d_{model}} \\
\\
W_{OV} \;&=\; W_V W_O \; &&\in \mathbb{R}^{d_{model} \times d_{model}} \\
W_{OV,\text{ full}} \;&=\; W_{EE} W_{OV} W_U \; &&\in \mathbb{R}^{d_{vocab} \times d_{vocab}} \\
\end{alignat*}
$$

with $W_U$ as the model's unembedding matrix, and $W_{EE}^T$ as the **extended embedding matrix** (see appendix).

<details>
<summary>Click on this dropdown for a more detailed mathematical justification.</summary>

### QK circuit

Suppose we are predicting token $T$ at destination position $D$, i.e. the query-side residual stream vector is $\left(W_U\right)_{[:, T]}$. Then the query vectors (writing them as column vectors) are:

$$
q = \left(W_Q\right)^T \left(W_U\right)_{[:, T]} = \left(W_Q^T W_U\right)_{[:, T]}
$$

and the key vector at source token $S$ is:

$$
k = \left(W_K\right)^T \left(W_{EE}\right)_{[S, :]} = \left(W_K^T W_{EE}\right)_{[S, :]}
$$

Note that we're making several simplifying assumptions here - in particular, assuming that the only signal which isn't sent to zero by the query matrix are the unembeddings, and that the only signal which isn't sent to zero by the key matrix are the (extended) embeddings.

Putting these two together, we find that the attention from $D$ to $S$ is:

$$
q^T k = \left(W_U^T W_Q W_K^T W_{EE}\right)_{T, S}
$$

and this attention should be largest when $S=T$ (which is what causes token $D$ to attend back to $S$ rather than some other token).

### OV circuit

Here, the story is similar. If the vector at source position $S$ is the extended embedding $\left(W_{EE}\right)_{[S, :]}$, then the value vector which gets moved to any token $D$ that pays attention to $S$ is:

$$
v = W_O^T W_V^T \left(W_{EE}\right)_{[S, :]}
$$

and the effect that this vector has on the model's predictions of token $T$ at the destination position is:

$$
v^T \left(W_U\right)_{[:, T]} = \left(W_{EE} W_V W_O W_U\right)_{[S, T]}
$$

We expect that attending to $S$ causes the prediction of $S$ to be suppressed, in other words that the above quantity is most negative when $T=S$.

</details>

<br><br>

## Interactive histograms

You can test this below, by choosing:

* Source token $S$ (i.e. we assume $(W_{EE})_{[S,:]}$ is at the source position), and
* Predicted token $T$ (i.e. we assume $(W_U)_{[:, T]}$ is at the destination position).

You should observe that the QK/OV circuit is most positive/negative respectively when $S=T$ (as well as a few other interesting patterns, which we'll discuss below).

<br>
""", unsafe_allow_html=True)



def format_user_input(s: str):
    s = s.strip()
    for char in ["'", '"']:
        if s.startswith(char) and s.endswith(char):
            s = s[1:-1]
            break
    else:
        with error_box[0]:
            st.error(f"One of your tokens isn't wrapped in quotes: [{s!r}]. You should use single or double quotes to wrap all your tokens.")
            return
    return s

@st.cache_data(show_spinner=False, max_entries=1)
def create_histograms(
    src_input: str,
    dest_input: str,
    k: int,
    focus_on: Literal["source", "destination"]
):
    # Split by commas
    src_input_split = src_input.strip(" ,").split(",")
    dest_input_split = dest_input.strip(" ,").split(",")
    # Remove the whitespace from each token, and make sure it's wrapped with quotations
    # Also check if any of them returned "None", in which case we should return None
    src_str_toks = list(map(lambda x: format_user_input(x.strip()), src_input_split))
    if None in src_str_toks: return None, None
    dest_str_toks = list(map(lambda x: format_user_input(x.strip()), dest_input_split))
    if None in dest_str_toks: return None, None
    # Check if any are just whitespace, e.g. if input was {" pier",} then this might have accidentally created whitespace

    tokenizer = dict_to_store_less["tokenizer"]

    for s, msg in zip(src_str_toks + dest_str_toks, ["source" for _ in src_str_toks] + ["destination" for _ in dest_str_toks]):
        _s = s if not s.startswith(" ") else "Ġ" + s[1:]
        if _s not in tokenizer.vocab:
            with error_box: st.error(error_msg.format(s=s, msg=msg))
            return None, None

    hist_QK = plot_full_matrix_histogram(
        dict_to_store_less,
        src=src_str_toks,
        dest=dest_str_toks,
        k=k,
        circuit="QK",
        neg=False,
        head="10.7",
        flip=(focus_on=="source"),
    )
    if hist_QK is None: return None, None

    hist_OV = plot_full_matrix_histogram(
        dict_to_store_less,
        src=src_str_toks,
        dest=dest_str_toks,
        k=k,
        circuit="OV",
        neg=True,
        head="10.7",
        flip=(focus_on=="destination")
    )
    return hist_QK, hist_OV


# Function to tell the Streamlit page that the histograms should be displayed, with params that will be defined below.
def queue_up_histogram():
    st.session_state["waiting_to_display"] = True
# We also make sure that the histogram runs the first time the page is loaded
if "waiting_to_display" not in st.session_state: 
    queue_up_histogram()


# Widgets for defining histogram params. When these are changed, the page is reloaded, and the code to display the histograms is below this.
cols = st.columns(2)
with cols[1]:
    k = st.slider("Number of top tokens to show: ", min_value=5, max_value=30, value=15, key="k", on_change=queue_up_histogram)
    focus_on = st.radio("Which token to focus on?", ["source", "destination"], key="focus_on", on_change=queue_up_histogram)
with cols[0]:
    src_input = st.text_input("Source token(s)", "' token'", key="src_input", on_change=queue_up_histogram)
    dest_input = st.text_input("Destination token(s)", "' token'", key="dest_input", on_change=queue_up_histogram)

error_box = st.container()
st.markdown("<br>", unsafe_allow_html=True)


# We run the histograms with params from session state (which exist because of the "key" parameters above).
cols2 = st.columns(2)
if st.session_state.get("waiting_to_display", False):
    st.session_state["waiting_to_display"] = False
    hist_QK, hist_OV = create_histograms(st.session_state.src_input, st.session_state.dest_input, st.session_state.k, st.session_state.focus_on)
    if hist_QK is not None:
        with cols2[0]: st.plotly_chart(hist_QK, use_container_width=True)
        with cols2[1]: st.plotly_chart(hist_OV, use_container_width=True)

st.markdown(
r"""
<br><br>

## Some interesting patterns to note

### Function words

Function words like `" the"`, `" of"`, `" where"` etc, are a class of words for which these results don't hold up. In other words, if we predict one of these words that doesn't mean we'll attend back to previous instances, and if we attend back to previous instances that doesn't mean we'll suppress that word's prediction.

Our current theory for why this happens is as follows:

> This negative head formed in order to suppress naive copying behaviour from earlier parts of the model. But with function words it's less useful to learn to copy them (because their appearance in a prompt shouldn't increase the probability that they'll appear later in the same prompt). So with these words, no naive copying behaviour was learned in the first place, and so the negative heads didn't learn to suppress it.

However, this is very speculative, and it's still an open question!

### Semantic similarity

You might have found a theme in what words tokens will most attend to, and what words will be most suppressed when you attend to a particular token. In particular, you might have found:

* Plurals (e.g. `" device"` attends back to & suppresses `" devices"`)
* Capitals (e.g. `" pier"` attends back to & suppresses `" Pier"`)
* Spaces (e.g. `"head"` attends back to & suppresses `"Head"`)
* Tokenization weirdness (e.g. `" Berkeley"` attends back to & suppresses `"keley"`)

What's going on here?

A naive theory would be that this effect just happens because the cosine similarity of the unembedding vectors is large. While this is true, it's not the whole story (e.g. the cosine similarity of most words' unembeddings with their capitalized unembeddings is usually around 0.5, but from the histograms above you should find that the logit suppression scores are often about equal for the word and its capitalization).

Here's a slightly more complex theory - we can imagine that the model has different circuits for performing:

* **Context details** (e.g. "this word has something to do with `"device"`)
* **Grammatical details** (e.g. whether the word is plural or starts with a space)

The key point here is that is that **copy-suppression isn't limited to pure copying, it also applies to context details.** If we attend back to `" Pier"`, this should suppress the probability of both `" pier"` and `" Pier"` at our destination token (because the copy-suppression circuit isn't the one that "knows" the grammatical details). You can see this exact effect in prompt 36, on the "Browse Examples" page.


""", unsafe_allow_html=True)

# We've observed that this is approximately a symmetric relation, i.e. the following statements are all closely related:
# * The unembedding for token $T$ attends back to the token $S$ (i.e. QK circuit).
# * The unembedding for token $S$ attends back to the token $T$ (i.e. QK circuit).
# * The token $S$ suppresses the token $T$ when it is attended to (i.e. OV circuit).
# * The token $T$ suppresses the token $S$ when it is attended to (i.e. OV circuit).
# * $S$ and $T$ are semantically related, in the sense that they both "point towards the same root word", they just differ by some grammatical details.

st.markdown(
r"""
<br><br>

### Appendix - extended embedding matrix

We define the extended embedding matrix $W_{EE}$ as follows:

* Take the "raw embeddings" from $W_E$,
* Apply the zeroth attention layer $\text{Attn-0}$ (fixing each head's self-attention to be 1),
* Apply $\text{MLP-0}$.

This is because GPT2-small essentially uses $\text{MLP-0}$ as an extended embedding matrix, to get around the restriction that the embedding and unembedding matrices are tied (so they aren't free to model bigrams, since bigrams are not a symmetric relation). We validated this choice my measuring the average cross entropy loss when self-attention for all heads in layer zero is set to 1 - the average increase in cross-entropy loss was about 1.0 (much smaller than the roughly 4.0 average loss from ablating these heads).

""", unsafe_allow_html=True)