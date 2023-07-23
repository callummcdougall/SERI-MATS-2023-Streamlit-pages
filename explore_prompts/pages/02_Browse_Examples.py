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

import streamlit as st
st.set_page_config(layout="wide")
from streamlit.components.v1 import html
from pathlib import Path
import pickle
import gzip

from streamlit_styling import styling
from generate_html import CSS
from explore_prompts_utils import ST_HTML_PATH

import torch as t
t.set_grad_enabled(False)

styling()

@st.cache_data(show_spinner=False, max_entries=1)
def load_html():
    with gzip.open(ST_HTML_PATH / "GZIP_HTML_PLOTS.pkl", "rb") as f:
        HTML_PLOTS = pickle.load(f)
    return HTML_PLOTS

HTML_PLOTS = load_html()

BATCH_SIZE = len(HTML_PLOTS["LOGITS_ORIG"])

NEG_HEADS = ["10.7", "11.10"]
EFFECTS = ["direct", "indirect", "both"]
LN_MODES = ["frozen", "unfrozen"]
ABLATION_MODES = ["mean", "zero"]
ATTENTION_TYPES = ["info-weighted", "standard"]
ATTN_VIS_TYPES = ["large", "small"]

first_idx = 36
batch_idx = st.sidebar.slider("Pick a sequence", 0, BATCH_SIZE-1, first_idx)
head_name = st.sidebar.radio("Pick a head", NEG_HEADS) # , "both"
assert head_name != "both", "Both not implemented yet. Please choose either 10.7 or 11.10"
effect = st.sidebar.radio("Pick a type of intervention effect", EFFECTS)
ln_mode = st.sidebar.radio("Pick a type of layernorm mode for the intervention", LN_MODES)
ablation_mode = st.sidebar.radio("Pick a type of ablation", ABLATION_MODES)
full_ablation_mode = "+".join([effect, ln_mode, ablation_mode])
full_ablation_mode_dla = "+".join([ln_mode, ablation_mode])

st.sidebar.markdown(r"""
---

### Explanation for the sidebar options

When we intervene on an attention head, there are 3 choices we can make. We can choose the:

* **Intervention effect** - just the direct path from the head to the final logits, or just the indirect path (everything excluding the direct path), or both.
* **LayerNorm mode** - whether we freeze the LayerNorm parameters to what they were on the clean run, or unfreeze them (so the new residual stream is normalised).
* **Ablation type** - we can mean-ablate, or zero-ablate.
""")

# HTML_LOSS = HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_mode)]
HTML_LOGITS_ORIG = HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)]
HTML_LOGITS_ABLATED = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, full_ablation_mode)]
# HTML_DLA = HTML_PLOTS["DLA"][(batch_idx, head_name, "NEG/POS")]
# HTML_ATTN = HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, attn_type)]
# HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)]

st.markdown(
r"""# Browse Examples

### What is this page for?

This page allows you to browse through a large set of OpenWebText prompts, and see how negative heads (10.7 and 11.10) behave on the different tokens in those prompts. It should help build your intuition about how much of the negative heads' behaviour is copy-suppression, and when this is good/bad.

### How does each visualisation relate to copy-suppression?

* **Loss difference from ablating negative head** - to figure out when a head is important, we need to see when ablating it changes the loss. We offer 4 different kinds of ablation: zero vs mean ablation, and removing direct effect vs patching (the latter includes indirect effects).
* **Direct effect on logits** - in situations where the heads are important, we need to know what tokens are being pushed up/down by this head, and how the logit output changes when we ablate the head.
* **Attention patterns** - hopefully the direct effect plot shows that, in most important examples, 10.7 is suppressing a token which appears earlier in context. The attention patterns should show that 10.7 is also attending back to this token (i.e. it's being negatively copied).
* **Prediction-attention?** - hopefully the attention plot shows that we attend back to the token which is being suppressed, but why do we do this? Our theory states that the unembedding of this token is a large component query-side. This plot shows the size of the unembedding component of the query-side attention for the token it's attending to, relative to the average size for other tokens in the context.

When all put together, the 2nd/3rd/4th plots should show that:

* Negative heads' direct effect on logits is to push down a token which appears earlier in context,
* Negative heads are attending to this token which appears earlier in context,
* The reason they're attending to this token is because it was being predicted at that destination position.

<br>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Loss",
    "Logits",
    "Attention Patterns",
    "Prediction-Attention?",
])

with tabs[0]:
    st.markdown(
r"""
<style>
mark {
    font-size: 1rem;
    line-height: 1.8rem;
    padding: 1px;
    margin-right: 1px;
}
</style>

## Loss difference from ablating negative head

This visualisation shows the loss difference from ablating head 10.7 (for various different kinds of ablation).

The sign is (loss with ablation) - (original loss), so <mark style="background-color:rgb(99,167,206)">&nbsp;blue</mark> (positive) means the loss increases when you ablate, i.e. the head is useful. <mark style="background-color:rgb(231,135,107)">&nbsp;Red</mark> means the head is harmful.

*You can hover over a token to see the loss difference, and you can click on a token to lock it. You can lock multiple tokens at once, if you want to compare them.*

<details>
<summary>Analysis</summary>

If we want to answer "what is a particular head near the end of the model doing, and why is it useful?" then it's natural to look at the cases where ablating it has a large effect on the model's loss.

Our theory is that **negative heads detect tokens which are being predicted (query-side), attend back to previous instances of that token (key-side), and negatively copy them, thereby suppressing the logits on that token.** The 2 plots after this one will provide evidence for this.

We'll use as an example the string `"...whether Bourdain Market will open at the pier"` (sequence **#36**), specifically the `"at the pier"` part. But we think these results hold up pretty well in most cases where ablating a head has a large effect on loss (i.e. this result isn't cherry-picked).

</details>

<br>

""", unsafe_allow_html=True)
    
    max_loss_color_is_free = st.checkbox("By default, the extreme colors are ± 2.5 cross-entropy loss. Check this box to extremise the colors (so the extreme color is the largest-magnitude change in loss in this sequence).")

    HTML_LOSS = HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_mode, max_loss_color_is_free)]
    html(CSS.replace("min-width: 275px", "min-width: 130px") + HTML_LOSS, height=400)

with tabs[1]:

    inner_tabs = st.tabs(["Direct logit attribution of head", "Logprobs before/after ablation"])

    with inner_tabs[0]:

        st.markdown(
r"""
## Direct logit attribution of head

Hover over token T to see what the direct effect of 10.7 is on the logits for T (as a prediction).

Note - to interpret this, we strongly recommend using mean ablation rather than zero-ablation, because it changes the baseline which logit attribution is measured with respect to.

<details>
<summary>Analysis</summary>

The notable observation - **for most of the examples where ablating 10.7 has a large effect on loss, you can see from here that they're important because they push down a token by a lot which appeared earlier in context.**

Take our `at the pier` example. The head is pushing down the prediction for `pier` by a lot, both following `at` and following `the`. In the first case this is helpful, because `pier` actually didn't come next. But in the second case it's unhelpful, because `pier` did come next.

In contrast, if you look at the tokens with very positive DLA, they're usually uninterpretable. **This head is better understood as pushing tokens down than as pushing tokens up.**

To complete this picture, we still want to look at the attention patterns, and verify that the head is attending to the token it's pushing down on. Note that, to make the logits more interpretable, I've subtracted their mean (so they're "logits with mean zero" not "logprobs").

</details>

<br>
""", unsafe_allow_html=True)

        radio_negpos_logits = st.radio(
            label = "For each token, view the tokens with the most...", 
            options = ["neg", "pos"],
            format_func = lambda x: {
            "neg": "NEGATIVE direct logit attribution",
            "pos": "POSITIVE direct logit attribution",
        }[x])

        HTML_DLA = HTML_PLOTS["DLA"][(batch_idx, head_name, full_ablation_mode_dla, radio_negpos_logits)]
        html(CSS + HTML_DLA, height=600)

    with inner_tabs[1]:
        st.markdown(
r"""
## Logprobs before/after ablation

In the columns below, you can also compare whole model's predictions before / after ablation.

The left is colored by probability assigned to correct token. The right is colored by change in logprobs assigned to correct token relative to the left (so red means ablation decreased the logprobs, i.e. the head was helpful here).

<details>
<summary>Analysis</summary>

You can see, for instance, that the probability the model assigns to ` Pier` following both the `at` and `the` tokens increases by a lot when we ablate head 10.7. For `(at, the)`, the `Pier`-probability goes from 63.80% to 93.52% when we ablate, which pushes the `the`-probability down (which in this case is bad for the model), and for `(the, pier)`, the `pier`-probability goes from 0.86% to 2.41% (which in this case is good for the model).

Note that the negative head is suppressing both `pier` and `Pier`, because they have similar embeddings. As we'll see below, it's actually suppressing `Pier` directly, and suppressing `pier` is just a consequence of this.

</details>


""", unsafe_allow_html=True)

        cols = st.columns(2)

        with cols[0]:
            st.markdown("### Original")
            st.caption("Colored by logprob assigned to correct next token.")
            html(CSS + HTML_LOGITS_ORIG, height=900)
        with cols[1]:
            st.markdown("### Ablated")
            st.caption("Colored by change in logprob for correct next token.")
            html(CSS + HTML_LOGITS_ABLATED, height=900)

with tabs[2]:
    st.markdown(
r"""
## Attention patterns

The visualisations below show what each token is attending to. You can change the attention mode to be info-weighted, meaning we scale each attention probability as follows:

$$
A^h[s_Q, s_K] \times \frac{\Big\|v^h[s_K]^T W_O^h\Big\|}{\underset{s}{\max} \Big\|v^h[s]^T W_O^h\Big\|}
$$

because this is a better representation of actual information flow.

<details>
<summary>Analysis</summary>

We expect to see both `at` and `the` attending back to `Pier` in sequence #36 - this is indeed what we find.

Note that the other two clear examples of a nontrivial attention pattern in this example also seem like they could be examples of "unembedding of token T attending to previous embedding of token T":

* `questioning` and `whether` attend to `Anthony` and `Bour`
    * It seems reasonable that `Anthony` and `Bour` are being predicted at these points. From the first plot, we can see that this is bad in the `questioning` case, because `Bour` was actually correct.
* `Eater` attends to `NY`
    * This is probably induction-suppression (i.e. `NY` was predicted because of induction, and this forms 10.7's attn pattern). In this case, it's good, because `NY` didn't come first.

</details>

<br><hr>
""", unsafe_allow_html=True)

    attn_type_checkbox = st.checkbox("Info-weighted attention?", value=True)
    attn_type = "info-weighted" if (attn_type_checkbox == True) else "standard"
    vis_name = st.radio("Pick a type of attention view", ["Small", "Large"])

    HTML_ATTN = HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, attn_type)]
    html(HTML_ATTN, height=1500)

with tabs[3]:
    st.markdown(
r"""
## Prediction-attention?

Our theory for copy-suppression is that each destination token `D` will attend back to `S` precisely when `S` is being predicted to follow `D`. To test this, we use the logit lens, i.e. we see what the model is predicting at each sequence position just before head """ + head_name + r""". As well as the top 10, we also include in this dropdown the largest unembedding components for words in the sequence.

We expect to see that, if a token `D` attends strongly to `S`, this means `S` (or a semantically similar token) is being predicted at `D`.

<details>
<summary>Analysis</summary>

We can see that there are no "false negatives" in sequence #36:

* `' at'` and `' the'` both have large components in the `' Pier'` direction, explaining these attention patterns.
* `' Eater'` has a large component in the `' NY'` direction, explaining this attention pattern.

But there are also a few "false positives", e.g. the second `' Bour'` token doesn't seem to be attending back to the first `'dain'` token despite this being strongly predicted.

We still don't know exactly why false positives occur - one possible theory is that the copy-suppression machinery has a way of "switching off" for certain kinds of situations, e.g. bigrams, when there's less risk of the model being overconfident. The negative head's default behaviour is to attend back to the BOS token (or whatever the first token happens to be), so maybe there's a component which writes strongly in the direction which causes this to happen.
</details>
""", unsafe_allow_html=True)
    # remove_self = st.checkbox("Remove self-attention from possible source tokens", value=False)

    HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)]
    html(CSS + HTML_UNEMBEDDINGS, height=1500)
