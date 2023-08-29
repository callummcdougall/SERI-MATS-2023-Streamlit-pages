import sys, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

# Reason we need this code: if we don't have it, then we default to importing the version of transformer_lens from site-packages instead
# (please correct me if wrong!)

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

import platform
is_local = (platform.processor() != "")

st.set_page_config(layout="wide")

from transformer_lens.rs.callum2.utils import ST_HTML_PATH

if is_local:
    # NEGATIVE_HEADS = [(10, 1), (10, 7), (11, 10)]
    NEGATIVE_HEADS = [(10, 7)]
    FILENAME = "OV_QK_circuits_less.pkl"
else:
    NEGATIVE_HEADS = [(10, 7)]
    FILENAME = "OV_QK_circuits_less.pkl"

NEG_HEADS = [f"{layer}.{head}" for layer, head in NEGATIVE_HEADS]


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
    k: int,
    neg: bool,
    circuit: Literal["QK", "OV"],
    focus_on: Literal["source", "destination"],
):
    '''
    By default, this looks at what dest most attends to (QK) or what src causes to be most suppressed (OV).

    But if "flip" is True, then it looks at what things attend to src most (OV), or what causes dest to be most suppressed (OV).
    '''
    W_EE_V = dict_to_store_less["W_EE_V"]
    W_U_O = dict_to_store_less["W_U_O"]
    W_U_Q = dict_to_store_less["W_U_Q"]
    W_EE_K = dict_to_store_less["W_EE_K"]

    # denom = (model.cfg.d_head ** 0.5)
    denom = 64 ** 0.5

    src_toks = tokenizer(src, return_tensors="pt")["input_ids"].squeeze().tolist()
    dest_toks = tokenizer(dest, return_tensors="pt")["input_ids"].squeeze().tolist()
    if isinstance(src, str): src = [src]
    if isinstance(src_toks, int): src_toks = [src_toks]
    if isinstance(dest, str): dest = [dest]
    if isinstance(dest_toks, int): dest_toks = [dest_toks]

    if circuit == "OV":
        if focus_on == "destination":
            if len(dest_toks) > 1:
                st.error(f"Error: if you're focusing on the destination token, you should only have a single destination token in the input. Instead you have {dest}.{short_error_msg}")
                return
            hist_toks = src_toks
            # W_U_O_toks_scaled = W_U_O.T[dest_toks[0]] / W_U_O.T[dest_toks[0]].std(dim=-1, keepdim=True)
            # W_EE_V_scaled = W_EE_V / W_EE_V.std(dim=-1, keepdim=True)
            # full_vector: Float[Tensor, "d_vocab"] = W_EE_V_scaled @ W_U_O_toks_scaled
            full_vector: Float[Tensor, "d_vocab"] = W_EE_V @ W_U_O[:, dest_toks[0]]
        else:
            if len(src_toks) > 1:
                st.error(f"Error: if you're focusing on the source token, you should only have a single source token in the input. Instead you have {src}.{short_error_msg}")
                return
            hist_toks = dest_toks
            # W_EE_V_toks_scaled = W_EE_V[src_toks[0]] / W_EE_V[src_toks[0]].std(dim=-1, keepdim=True)
            # W_U_O_scaled = W_U_O / W_U_O.std(dim=-1, keepdim=True)
            # full_vector: Float[Tensor, "d_vocab"] = W_EE_V_toks_scaled @ W_U_O_scaled
            full_vector: Float[Tensor, "d_vocab"] = W_EE_V[src_toks[0]] @ W_U_O

        full_vector_topk = full_vector.topk(k, dim=-1, largest=not(neg))

    elif circuit == "QK":
        if focus_on == "source":
            if len(src_toks) > 1:
                st.error(f"Error: if you're focusing on the source token, you should only have a single source token in the input. Instead you have {src}.{short_error_msg}")
                return
            hist_toks = dest_toks
            # W_EE_K_toks_scaled = W_EE_K[src_toks[0]] / W_EE_K[src_toks[0]].std(dim=-1, keepdim=True)
            # W_U_Q_scaled = W_U_Q / W_U_Q.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_U_Q @ W_EE_K[src_toks[0]] / denom
        else:
            if len(dest_toks) > 1:
                st.error(f"Error: if you're focusing on the destination token, you should only have a single destination token in the input. Instead you have {dest}.{short_error_msg}")
                return
            hist_toks = src_toks
            # W_EE_scaled = W_EE_K / W_EE_K.std(dim=-1, keepdim=True)
            # W_U_Q_toks_scaled = W_U_Q[dest_toks[0]] / W_U_Q[dest_toks[0]].std(dim=-1, keepdim=True)
            # full_vector: Float[Tensor, "d_vocab"] = W_U_Q_toks_scaled @ W_EE_scaled.T / denom
            full_vector: Float[Tensor, "d_vocab"] = W_U_Q[dest_toks[0]] @ W_EE_K.T / denom

        full_vector_topk = full_vector.topk(k, dim=-1, largest=not(neg))
    
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
    
    title = f"<br><b style='font-size:22px;'>{circuit} circuit</b>:<br>"
    if (circuit, focus_on) == ("OV", "destination"):
        title += f"Which source tokens most {'suppress' if neg else 'boost'} the prediction of <b>{dest[0].replace(' ', 'Ġ')!r}</b> ?" # customwrap(
    elif (circuit, focus_on) == ("OV", "source"):
        title += f"Which predictions does source token <b>{src[0].replace(' ', 'Ġ')!r}</b> {'suppress' if neg else 'boost'} most, when attended to?"
    elif (circuit, focus_on) == ("QK", "source"):
        title += f"Which tokens' unembeddings {'least' if neg else 'most'} attend to source token <b>{src[0].replace(' ', 'Ġ')!r}</b> ?"
    elif (circuit, focus_on) == ("QK", "destination"):
        title += f"Which source tokens does the unembedding of <b>{dest[0].replace(' ', 'Ġ')!r}</b> attend to {'least' if neg else 'most'}?"

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
        df, x="y", y="x", color="color", template="simple_white", title=title.replace("Ġ", " "),
        width=650, height=250+25*len(x), labels={"y": "Logits" if (circuit=="OV") else "Attention score", "x": "", "color": "Token class"},
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

@st.cache_data(show_spinner=False, max_entries=1)
def get_mega_dict() -> dict:
    mega_dict = pickle.load(gzip.open(ST_HTML_PATH / FILENAME))
    # Get it in the same form as the version which has multiple heads, if this is the one with only 10.7
    if "10.7" not in mega_dict:
        mega_dict = {
            "tokenizer": mega_dict["tokenizer"],
            "10.7": {k: v for k, v in mega_dict.items() if k != "toknizer"},
        }
    return mega_dict


mega_dict = get_mega_dict()
tokenizer = mega_dict.pop("tokenizer")

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
    focus_on: Literal["source", "destination"],
    head_name: str,
    flipped: bool,
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

    for s, msg in zip(src_str_toks + dest_str_toks, ["source" for _ in src_str_toks] + ["destination" for _ in dest_str_toks]):
        _s = s if not s.startswith(" ") else "Ġ" + s[1:]
        if _s not in tokenizer.vocab:
            with error_box: st.error(error_msg.format(s=s, msg=msg))
            return None, None

    hist_QK = plot_full_matrix_histogram(
        mega_dict[head_name],
        src=src_str_toks,
        dest=dest_str_toks,
        k=k,
        neg=False,
        circuit="QK",
        focus_on=focus_on,
    )
    if hist_QK is None: return None, None

    hist_OV = plot_full_matrix_histogram(
        mega_dict[head_name],
        src=src_str_toks,
        dest=dest_str_toks,
        k=k,
        neg=True if not(flipped) else False,
        circuit="OV",
        focus_on=focus_on,
    )
    return hist_QK, hist_OV


# Function to tell the Streamlit page that the histograms should be displayed, with params that will be defined below.
def queue_up_histogram():
    st.session_state["waiting_to_display"] = True
# We also make sure that the histogram runs the first time the page is loaded
if "waiting_to_display" not in st.session_state: 
    queue_up_histogram()


# Widgets for defining histogram params. When these are changed, the page is reloaded, and the code to display the histograms is below this.
head_name = st.sidebar.radio("Pick a head", NEG_HEADS, key="head_name", on_change=queue_up_histogram)
cols = st.columns(2)
with cols[1]:
    k = st.slider("Number of top tokens to show: ", min_value=5, max_value=30, value=15, key="k", on_change=queue_up_histogram)
    focus_on = st.radio("Which token to focus on?", ["source", "destination"], key="focus_on", on_change=queue_up_histogram)
    if is_local:
        flipped = st.checkbox("Flip order of OV circuit?", key="flipped", on_change=queue_up_histogram)
    else:
        flipped = False
        st.session_state["flipped"] = flipped
with cols[0]:
    src_input = st.text_input("Source token(s)", "' token'", key="src_input", on_change=queue_up_histogram)
    dest_input = st.text_input("Destination token(s)", "' token'", key="dest_input", on_change=queue_up_histogram)

error_box = st.container()
st.markdown("<br>", unsafe_allow_html=True)


# We run the histograms with params from session state (which exist because of the "key" parameters above).
cols2 = st.columns(2)
if st.session_state.get("waiting_to_display", False):
    st.session_state["waiting_to_display"] = False
    hist_QK, hist_OV = create_histograms(
        src_input = st.session_state.src_input,
        dest_input = st.session_state.dest_input,
        k = st.session_state.k,
        focus_on = st.session_state.focus_on,
        head_name = st.session_state.head_name,
        flipped = st.session_state.flipped
    )
    if hist_QK is not None:
        with cols2[0]: st.plotly_chart(hist_QK, use_container_width=True)
        with cols2[1]: st.plotly_chart(hist_OV, use_container_width=True)
    else:
        st.error("noo")

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