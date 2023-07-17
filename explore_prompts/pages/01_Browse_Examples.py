# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os

try:
    root_dir = os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts"
    os.chdir(root_dir)
except:
    root_dir = "/app/seri-mats-2023-streamlit-pages/explore_prompts"
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

styling()

with gzip.open(ST_HTML_PATH / "GZIP_HTML_PLOTS.pkl", "rb") as f:
    HTML_PLOTS = pickle.load(f)

BATCH_SIZE = len(HTML_PLOTS["LOGITS_ORIG"])

NEG_HEADS = ["10.7", "11.10"]
ABLATION_TYPES = ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]
ATTENTION_TYPES = ["info-weighted", "standard"]
ATTN_VIS_TYPES = ["large", "small"]

first_idx = 36
batch_idx = st.sidebar.slider("Pick a sequence", 0, BATCH_SIZE, first_idx)
head_name = st.sidebar.radio("Pick a head", NEG_HEADS + ["both"])
assert head_name != "both", "Both not implemented yet. Please choose either 10.7 or 11.10"
ablation_type = st.sidebar.radio("Pick a type of ablation", ABLATION_TYPES)

HTML_LOSS = HTML_PLOTS["LOSS"][(batch_idx, head_name, ablation_type)]
HTML_LOGITS_ORIG = HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)]
HTML_LOGITS_ABLATED = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, ablation_type)]
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
""")

tabs = st.tabs([
    "Loss",
    "Logits",
    "Attention Patterns",
    "Prediction-Attention?",
])

with tabs[0]:
    st.markdown(
r"""
## Loss difference from ablating negative head

This visualisation shows the loss difference from ablating head 10.7 (for various different kinds of ablation).

The sign is (loss with ablation) - (original loss), so blue (positive) means the loss increases when you ablate, i.e. the head is useful. Red means the head is harmful.

### todo - increase the contrast on this plot, the colors are frustratingly muted. Maybe have a checkbox option to toggle between "2.5 = max color" and "max value = max color".

<details>
<summary>Analysis</summary>

If we want to answer "what is a particular head near the end of the model doing, and why is it useful?" then it's natural to look at the cases where ablating it has a large effect on the model's loss.

Our theory is that **negative heads detect tokens which are being predicted (query-side), attend back to previous instances of that token (key-side), and negatively copy them, thereby suppressing the logits on that token.** The 2 plots after this one will provide evidence for this.

We'll use as an example the string `"...whether Bourdain Market will open at the pier"`, specifically the `"at the pier"` part. But we think these results hold up pretty well in most cases where ablating a head has a large effect on loss (i.e. this result isn't cherry-picked).

</details>

""", unsafe_allow_html=True)

    html(CSS.replace("min-width: 275px", "min-width: 100px") + HTML_LOSS, height=200)

with tabs[1]:

    inner_tabs = st.tabs(["Direct logit attribution of head", "Logits before/after ablation"])

    with inner_tabs[0]:

        st.markdown(
r"""
## Direct logit attribution of head

Hover over token T to see what the direct effect of 10.7 is on the logits for T (as a prediction).

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

        HTML_DLA = HTML_PLOTS["DLA"][(batch_idx, head_name, radio_negpos_logits)]
        html(CSS + HTML_DLA, height=400)

    with inner_tabs[1]:
        st.markdown(
r"""
## Logits before/after ablation

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
            html(CSS + HTML_LOGITS_ORIG, height=800)
        with cols[1]:
            st.markdown("### Ablated")
            st.caption("Colored by change in logprob for correct next token.")
            html(CSS + HTML_LOGITS_ABLATED, height=800)

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

Our theory for copy-suppression is that each destination token `D` will attend back to `S` precisely when `S` is being predicted to follow `D`. To test this, we show (for each token `D` in the sequence) all the tokens `S` before it in context, such that the residual stream at `D` has a large component in the direction of `S`'s unembedding. We color each `D` by the maximum component over all `S` (i.e. the darker tokens are ones where a token which appeared earlier in context is being predicted with high probability).

> *Note - this visualisation might be redesigned, because I'm not sure this is the most elegant way to display this information. There's a few hacky things here, e.g. having to subtract the mean unembedding component over source tokens `S` for each destination token `D`, and the fact that `D` will often contain a nontrivial component of its own unembedding because of tied embeddings.*

<details>
<summary>Analysis</summary>

We can see that there are no "false negatives" in sequence #36:

* `' at'` and `' the'` both have large components in the `' Pier'` direction, explaining these attention patterns.
* `' Eater'` has a large component in the `' NY'` direction, explaining this attention pattern.

But there are also a few "false positives", most notably the `' Bour'` token, which has a large component in the `'dain'` direction (likely due to bigrams/trigrams for the first example, and bigrams/induction for the second and third examples). There is actually some copy-suppression on the second `' Bour'` token (you can see this in the direct logit attribution where `' dain'` is pushed down by 0.5 logits, but not on the loss plots because the model was already so confident in `' dain'` that its probability is still basically 100%). But not only is 0.5 logits much less than the copy-suppression for e.g. `' Eater'` ➔ `' NY'` (2.36 logits), but also there's no copy-suppression at all on the third `' Bour'` token, which is a much clearer example of a false positive.

Further analysis shows that the lack of copy-suppression is a consequence of the attention patterns. **Why do we have these false positives, where a token is predicted with overwhelmingly high probability but this doesn't cause either neg head to attend back to its previous occurrence?** This is an open question! A few speculative theories:

* This has something to do with the "perpendicular direction" in IOI
* Bigrams shouldn't be suppressed, so neg heads are instructed to attend to BOS instead (i.e. they're "turned off" generically)
</details>
"""
)
    # remove_self = st.checkbox("Remove self-attention from possible source tokens", value=False)

    HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)]
    html(CSS + HTML_UNEMBEDDINGS, height=1500)
