import sys, os
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

st.set_page_config(layout="wide")

from transformer_lens.rs.callum2.utils import ST_HTML_PATH
from transformer_lens.rs.callum2.st_page.streamlit_styling import styling
styling()

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
    focus_on: Literal["source", "destination"] = "source",
):
    '''
    By default, this looks at what dest most attends to (QK) or what src causes to be most suppressed (OV).

    But if "flip" is True, then it looks at what things attend to src most (OV), or what causes dest to be most suppressed (OV).
    '''
    _head = head.replace(".", "")
    tokenizer = dict_to_store_less["tokenizer"]
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
    
    title = f"<br><b style='font-size:22px;'>{circuit} circuit</b>:<br>"
    if (circuit, focus_on) == ("OV", "destination"):
        title += customwrap(f"Which source tokens most suppress the prediction of <b>{dest[0].replace(' ', 'Ġ')!r}</b> ?")
    elif (circuit, focus_on) == ("OV", "source"):
        title += customwrap(f"Which predictions does source token <b>{src[0].replace(' ', 'Ġ')!r}</b> suppress most, when attended to?")
    elif (circuit, focus_on) == ("QK", "source"):
        title += customwrap(f"Which tokens' unembeddings most attend to source token <b>{src[0].replace(' ', 'Ġ')!r}</b> ?")
    elif (circuit, focus_on) == ("QK", "destination"):
        title += customwrap(f"Which source tokens does the unembedding of <b>{dest[0].replace(' ', 'Ġ')!r}</b> attend to most?")

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
def get_dict_to_store_less():
    return pickle.load(gzip.open(ST_HTML_PATH / f"OV_QK_circuits_less.pkl", "rb"))

dict_to_store_less = get_dict_to_store_less()

@st.cache_data(show_spinner=False, max_entries=1)
def get_semantically_similar_dict():
    return pickle.load(open(ST_HTML_PATH.parent.parent / "cspa/cspa_semantic_dict_full.pkl", "rb"))

cspa_semantic_dict_full = get_semantically_similar_dict()

st.markdown(
r"""
# Semantic Similarity

Some of the copy suppression done by head 10.7 is what we might call "pure copy suppression", where the model is predicting token $s$ and destination position $d$, and attends back to a previous instance of this token $s$ in order to suppress it. But another part of the story is what we call **"semantic copy suppression"**, where the model is predicting token $s^*$ at $d$, using this to attend back to $s$, and suppressing $s^*$. When we say "semantically similar", we mean one of the following:

* Capitalization (e.g. `" pier"`, `" Pier"` and `" PIER"` are all related),
* Prepended spaces (e.g. `" token"` and `"token"` are related),
* Pluralization (e.g. `" device"` and `" devices"` are related),
* Other relationships which are harder to quantify, for example:
    * Verb forms, e.g. `" jump"`, `" jumped"`, `" jumping"`
    * Noun or adjective forms of the word, e.g. `[" execute", " execution"]` or `[" drive", " driver"]`
* Tokenization (e.g. `" Berkeley"` and `"keley"` are related, because `"Berkeley"` is tokenized into [`"Ber"`, `"keley"`]).

We've hardcoded semantic similarity rules, and we use it in our copy-suppression preserving ablation. Below, you can enter a single token, and see the following:

* All the semantically similar tokens to this one (according to our rules),
* The QK and OV circuit results for this token (just like on the first page of this app), with the semantically similar tokens highlighted in orange.

<details>
<summary>Some examples for you to try out</summary>

Our hardcoded rules work very well for `" token"`, because all the words with very high prediction-attention and suppression scores are the seven semantically related tokens, which are precisely the variants of `" token"` with space / no space, plural / no plural, and capital / no capital (the only one of these eight not included is `"Tokens"` because it isn't a token in our vocabulary).

Here are a few more examples where our rules work well:

* `" device"` - basiclly the same story as `" token"`.
* `" Berkeley"` - we capture the tokens `"Ber"` and `"keley"` which are commonly produced during tokenization (although note that we miss out `" Berk"`, which is presumably produced during some other version of tokenization which we didn't include).
* `" robot"` - we capture the standard plural / space / capital versions, but also the words derived from "robotic" (which we can see from these results are also suppressed).

Here are some examples where our hardcoded rules work less well, and explanations for why. Note that we've left function words off this list, because we don't expect the results to hold up for function words (for reasons we've discussed).

* `" Cairo"` - although we capture the stem `"airo"` which is useful, we're missing out on words related to **Egypt** which are all very copy-suppressed. This shows one weakness with our semantic similarity rules - although being semantically similar is one way (possibly the main way) for two tokens to be treated the same in the QK and OV circuits, it's not the only way. If the QK and OV circuits are in some sense "collapsing the differences between words, except for the main kinds of differences i.e. topic-related differences", then it makes sense that it would think these two words are similar.
* `" run"` - this is an interesting example. It works well in one direction (if we're predicting `" run"`, then our semantic similarity rules correctly identify the source tokens that we'll attend back to, and the source tokens that will suppress the `" run"` prediction). But it works less well in the other direction, i.e. when `" run"` is a source token (and unfortunately this is the direction we need it to work well in for our ablation method). 


</details>

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
    tok_input: str,
    topk: int,
    K_sem: int,
):
    # Get semantically similar tokens and flatten the list
    tok_input = format_user_input(tok_input)
    semantically_similar_str_toks = cspa_semantic_dict_full[tok_input]
    semantically_similar_str_toks = [i for j in semantically_similar_str_toks for i in j][:K_sem]

    tokenizer = dict_to_store_less["tokenizer"]

    _s = tok_input if not tok_input.startswith(" ") else "Ġ" + tok_input[1:]
    if _s not in tokenizer.vocab:
        with error_box: st.error(error_msg.format(s=tok_input, msg="source"))
        return None, None
    
    hists_dict = {
        ("QK", "source"): {"src": [tok_input], "dest": semantically_similar_str_toks},
        ("QK", "destination"): {"src": semantically_similar_str_toks, "dest": [tok_input]},
        ("OV", "source"): {"src": [tok_input], "dest": semantically_similar_str_toks},
        ("OV", "destination"): {"src": semantically_similar_str_toks, "dest": [tok_input]},
    }
    hists_dict = {
        (circuit, focus_on): plot_full_matrix_histogram(
            dict_to_store_less = dict_to_store_less,
            **kwargs,
            circuit = circuit,
            focus_on = focus_on,
            neg = (circuit == "OV"),
            head = "10.7",
            k = topk,
        )
        for (circuit, focus_on), kwargs in hists_dict.items()
    }
    return hists_dict, semantically_similar_str_toks



# Widgets for defining histogram params. When these are changed, the page is reloaded, and the code to display the histograms is below this.
cols = st.columns(3)
with cols[0]:
    tok_input = st.text_input("Token", "' token'", key="tok_input")
with cols[1]:
    topk = st.slider("Number of top tokens to show: ", min_value=5, max_value=30, value=20, key="topk")
    K_sem = st.slider("Max number of semantically similar tokens to show: ", min_value=1, max_value=15, value=10, key="K_sem")

error_box = st.container()
st.markdown("<br>", unsafe_allow_html=True)


# We run the histograms with params from session state (which exist because of the "key" parameters above).
cols2 = st.columns(2)
cols3 = st.columns(2)
hists_dict, semantically_similar_str_toks = create_histograms(
    tok_input = st.session_state["tok_input"],
    topk = st.session_state["topk"],
    K_sem = st.session_state["K_sem"],
)
if hists_dict is not None:
    s = "\n".join(list(map(repr, semantically_similar_str_toks)))
    with cols[2]:
        st.markdown(
f"""### Similar tokens

```c
{s}
```
""", unsafe_allow_html=True)
    with cols2[0]: st.plotly_chart(hists_dict[("OV", "source")], use_container_width=True)
    with cols3[0]: st.plotly_chart(hists_dict[("QK", "source")], use_container_width=True)
    with cols2[1]: st.plotly_chart(hists_dict[("OV", "destination")], use_container_width=True)
    with cols3[1]: st.plotly_chart(hists_dict[("QK", "destination")], use_container_width=True)

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