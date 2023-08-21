# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Stuff to make the page work on my local machine
from pathlib import Path
for p in [
    Path(r"C:\Users\calsm\Documents\AI Alignment\SERIMATS_23\seri_mats_23_streamlit_pages"),
    Path(r"/home/ubuntu/SERI-MATS-2023-Streamlit-pages"),
]:
    if os.path.exists(str_p := str(p.resolve())):
        os.chdir(str_p)
        if (sys.path[0] != str_p): sys.path.insert(0, str_p)
        break

import streamlit as st
st.set_page_config(layout="wide")
from streamlit.components.v1 import html
import pickle
import gzip

import platform
is_local = (platform.processor() != "")

from transformer_lens.rs.callum2.st_page.streamlit_styling import styling
from transformer_lens.rs.callum2.generate_st_html.generate_html_funcs import CSS
from transformer_lens.rs.callum2.utils import ST_HTML_PATH
from transformer_lens.rs.callum2.st_page.Home import (NEGATIVE_HEADS, HTML_PLOTS_FILENAME)

import torch as t
t.set_grad_enabled(False)

styling()

@st.cache_data(show_spinner=False, max_entries=1)
def load_html():
    # filename = f"GZIP_HTML_PLOTS_b{200 if is_local else 51}_s61.pkl"
    filename = HTML_PLOTS_FILENAME
    with gzip.open(ST_HTML_PATH / filename, "rb") as f:
        HTML_PLOTS = pickle.load(f)
    return HTML_PLOTS

HTML_PLOTS = load_html()

BATCH_SIZE = len(HTML_PLOTS["LOGITS_ORIG"])

NEG_HEADS = [f"{layer}.{head}" for layer, head in NEGATIVE_HEADS]
EFFECTS = ["direct", "indirect", "both"]
LN_MODES = ["frozen", "unfrozen"]
ABLATION_MODES = ["mean", "zero"]
ATTENTION_TYPES = ["info-weighted", "standard"]
ATTN_VIS_TYPES = ["large", "small"]

first_idx = 36
if is_local:
    batch_idx = st.sidebar.number_input(f"Pick a sequence (0-{BATCH_SIZE-1})", 0, BATCH_SIZE-1, first_idx)
else:
    batch_idx = st.sidebar.slider("Pick a sequence", 0, BATCH_SIZE-1, first_idx)
head_name = st.sidebar.radio("Pick a head", NEG_HEADS) # , "both"
# assert head_name != "both", "Both not implemented yet. Please choose either 10.7 or 11.10"
if head_name == "10.7":
    EFFECTS = ["direct", "indirect", "indirect (excluding 11.10)", "both"]
effect = st.sidebar.radio("Pick a type of intervention effect", EFFECTS)
ln_mode = st.sidebar.radio("Pick a type of layernorm mode for the intervention", LN_MODES)
# ablation_mode = st.sidebar.radio("Pick a type of ablation", ABLATION_MODES)
ablation_mode = "mean"
full_ablation_mode = "+".join([effect, ln_mode, ablation_mode])
full_ablation_mode_dla = "+".join([ln_mode, ablation_mode])

st.sidebar.markdown(
r"""---

### Explanation for the sidebar options

<details>
<summary>Intervention effect</summary>
<br>

**Direct effect** = just the effect that the head has on the final logit output.

**Indirect effect** = the effect of all paths from the head to the final logit output *except for* the direct path.

**Both** = the sum of the direct and indirect effects.

**Indirect (excluding 11.10)** = if we're looking at head 10.7, then it's informative to look at the indirect effect not counting the path from 10.7 to 11.10. This is because both these heads are doing copy-suppression, and copy-suppression is self-correcting (i.e. once you do it, you remove the signal that causes it to be done).
                    
</details>

<details>
<summary>LayerNorm mode</summary>

<br>

If "frozen", then we use the same layernorm parameters than we did in the clean run (so e.g. the direct effect will just be a linear function of the output of the head). If "unfrozen", then the layernom parameters are recomputed for the new ablated run.
                    
</details>

""", unsafe_allow_html=True)

# <details>
# <summary>Ablation type</summary>
# <br>
# We can zero-ablate the output of our head, or mean-ablate. Mean ablation is more principled, because it takes into account things like a constant bias term.
# </details>

# HTML_LOSS = HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_mode)]
HTML_LOGITS_ORIG = HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)]
HTML_LOGITS_ABLATED = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, full_ablation_mode)]
HTML_LOGITS_CSPA = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, "CSPA")]
# HTML_DLA = HTML_PLOTS["DLA"][(batch_idx, head_name, "NEG/POS")]
# HTML_ATTN = HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, attn_type)]
# HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)]

st.markdown(
r"""# Browse Examples

### What is this page for?

This page allows you to browse through a large set of OpenWebText prompts, and see how negative heads (10.7 and 11.10) behave on the different tokens in those prompts. It should help build your intuition about how much of the negative heads' behaviour is copy-suppression, and when this is good/bad.

### What is copy suppression?

Let $s$ and $d$ stand for the source and destination tokens respectively. The idealised form of copy suppression works as follows: token $s$ is being predicted at destination token $d$ before the negative head, which causes $d$ to attend back to $s$ and suppress the prediction of $s$. This is beneficial when $s$ is in fact incorrect (because the head will be pushing down incorrect predictions), and it's harmful when $s$ is in fact correct (because the head will be pushing down correct predictions).

There's one additional complicating factor, which we've called **semantic copy suppression**. We can also have token $s^*$ being predicted at $d$, where $s^*$ is **semantically related** to a source token $s$. By this, we mean that they are the same token up to small variations e.g. a prepended space, capitalization, plurals, other minor morphological changes e.g. verb tenses, or cursed features of tokenization (more on this later).

### How does each visualisation relate to copy-suppression?

* <u>**Loss**</u>. You can see the loss difference per token when you ablate the head (we offer different kinds of ablation). A large increase means this head was very useful at this token. We define **MIDS** (max-impact dataset samples) as the tokens which have the largest change in loss when ablating the head.
* <u>**Logits**</u>. You can see the head's largest direct effect (positive and negative), as well as the top predictions before and after ablation. *You should see that, for MIDS, the largest negative direct effect at token $d$ is on some token $s^*$ which is semantically related to a source token $s$.*
* <u>**Attention patterns**</u>. *You should see that, for MIDS, $d$ attends back to $s$.*
* <u>**Logit lens**</u>. This shows you the largest predictions before the head, both over the entire vocabulary and over the source tokens. *You should see that, for MIDS, $s^*$ is predicted at $d$.*

Finally, we have a form of ablation which tries to delete all information except for this described copy suppression mechanism. We call this **CSPA** (copy-suppression preserving ablation). You can see the results of this ablation in the "CSPA" tab, and verify that all this qualitative behaviour holds up when put it to the test.

<br>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Loss",
    "Logits",
    "Attention Patterns",
    "Logit Lens",
    "CSPA",
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

This visualisation shows the loss difference from ablating the attention head (for various different kinds of ablation).

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
        
        DLA_CONTAINER = st.container()


    with inner_tabs[1]:
        st.markdown(
r"""
## Logprobs before/after ablation

In the columns below, you can also compare whole model's predictions before / after ablation.

The original logits are colored by probability assigned to the correct token. The ablated logits are colored by change in logprobs assigned to the correct token (so these colors are just the reverse of the colors for the loss plot earlier).

<details>
<summary>Analysis</summary>

You can see, for instance, that the probability the model assigns to ` Pier` following both the `at` and `the` tokens increases by a lot when we ablate head 10.7. For `(at, the)`, the `Pier`-probability goes from 63.80% to 93.52% when we ablate, which pushes the `the`-probability down (which in this case is bad for the model), and for `(the, pier)`, the `pier`-probability goes from 0.86% to 2.41% (which in this case is good for the model).

Note that the negative head is suppressing both `pier` and `Pier`, because they have similar embeddings. As we'll see below, it's actually suppressing `Pier` directly, and suppressing `pier` is just a consequence of this.

</details>


""", unsafe_allow_html=True)

        LOGITS_CONTAINER = st.container()

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

with tabs[4]:
    st.markdown(
r"""
## Copy suppression-preserving ablation

We describe this ablation in mode detail in the "Copy Suppression Classification" page. To summarize:

> *For each source token $s$, we project its attention result vector $v_s^T W_O$ onto the unembeddings of its semantically similar tokens $s^*$. This is how we test that we understand the OV circuit (i.e. it directly pushes on the logits of tokens which it attends to).*
> 
> *For each of these pairs $(s, s^*)$, we also mean-ablate them unless $s^*$ is being predicted at the destination token $d$. This is how we test that we understand the QK circuit (i.e. it attends to things which are predicting).*

We've emphasised all tokens which are in the top or bottom 3% of loss-affecting examples. We've also colored them by how much of this loss effect is lost when you perform CSPA:

* Blue means the head **improves** model performance here. Amount of improvement captured by copy-suppression = <td>&nbsp;<mark style="background-color:rgb(247, 247, 247)"><font color="black"><b>0%</b></font></mark>, &nbsp;<mark style="background-color:rgb(209, 229, 240)"><font color="black"><b>20%</b></font></mark>, &nbsp;<mark style="background-color:rgb(146, 197, 222)"><font color="black"><b>40%</b></font></mark>, &nbsp;<mark style="background-color:rgb(67, 147, 195)"><font color="black"><b>60%</b></font></mark>, &nbsp;<mark style="background-color:rgb(33, 102, 172)"><font color="white"><b>80%</b></font></mark>, &nbsp;<mark style="background-color:rgb(5, 48, 97)"><font color="white"><b>100%</b></font></mark></td>
* Red means the head **harms** model performance here. Amount of deterioration captured by copy-suppression = <td>&nbsp;<mark style="background-color:rgb(247, 247, 247)"><font color="black"><b>0%</b></font></mark>, &nbsp;<mark style="background-color:rgb(253, 219, 199)"><font color="black"><b>20%</b></font></mark>, &nbsp;<mark style="background-color:rgb(244, 165, 130)"><font color="black"><b>40%</b></font></mark>, &nbsp;<mark style="background-color:rgb(214, 96, 77)"><font color="black"><b>60%</b></font></mark>, &nbsp;<mark style="background-color:rgb(178, 24, 43)"><font color="white"><b>80%</b></font></mark>, &nbsp;<mark style="background-color:rgb(103, 0, 31)"><font color="white"><b>100%</b></font></mark></td>

In other words, we want to see very dark colors for all the bold words, because this shows that our understanding of copy suppression does match what the head is doing here.

<details>
<summary>Analysis - a sample of cases where CSPA fails to capture the effect, and why it fails.</summary>

Asterisk = it's sad that we missed this example because it does seem like copy suppression, and CSPA should be changed so that we get this example.

Hat = this is a nice example of something we missed, which it's okay that we missed.

This is also for us during research - at the end, all of these should make sense and be satisfying (without any asterisks). That'll be a good sign that we're done here.

#### (32) \n\n A ➔ course

Turns out that `" course"` and `" curriculum"` are top pairs in the QK circuit, so it makes sense that a prediction of `" curriculum"` would attend to the source token `" course"`, even if that doesn't show up in our model of copy-suppression. Not sure how to fix this, except to better understand exactly why words like these two are seen as equivalent by the model.

#### (40) Cairo talks continue between ➔ Israel

The head is useful here, because when we attend to `" Cairo"` we suppress `" Egypt"` (this can be seen from the OV circuit, and also it makes sense because they're related words). Our hardcoded semantic similarity doesn't cover this case. Note, the fact that CSPA fails here shows that `" Cairo"` suppressing `" Egypt"` isn't just a consequence of cosine similarity between these 2 words. From head 10.7's perspective, these tokens are very similar.

There are also a few other examples in (40) where CSPA fails for this same reason.

#### (51) condenm the senseless ➔ motorcycle

This seems like "distributed copy suppression". Our model captures about 50% of it, and the other 50% is because a few related tokens e.g. `" fatalities"` are also suppressed, and we don't capture that - that's fine.

#### (53) Both examples

There are only 3 examples here. None of them are captured very well. This is understandable, because things are quite distributed and not crisp.

#### (57^) Kanye (the comma one at the end)

We have attention to Kanye West and Taylor Swift (all four names). The head is harmful in this case because `" Kanye"` (the correct token) is suppressed. But we mostly attend to `" West"`. Interestingly, the OV and QK circuits don't think these words are related at all, so my guess is that something bigram-y is going on here (e.g. there's a neuron which activates on the full name Kanye West, and it's this neuron's output which is the important key-side and value-side component).

#### (73^)

Attention is incredibly distributed, so it's not highly localized. I think doing the non-sparse projection would perform better here, but I still think making it sparse is worth it.

#### (80^) Jihad vs . ➔ Mc

The head attends to `" Jihad"`, and pushes down - ahem - "related words", in particular the word `" Terror"` (as well as `" Islam"`, although to a lesser extent). This won't be captured by the narrow semantic similarity.

---

Many of the remaining examples make me think that these metrics would all look much crisper if I filtered for the most extreme 2.5% (or even 3-4%) rather than 5%. Lots of the unsatisfying cases are also borderline on 5% (and barely have any nontrivial attention, it just happens to be working in the right direction).

</details>

""", unsafe_allow_html=True)

    if is_local:
        show_cspa = st.checkbox("Tick this box to show the CSPA plots for logits & DLA on the second tab, using head 10.7, direct effect, frozen layernorm. *(This is mainly useful during research, so we can see where & why CSPA is failing to capture the effect.)*", value=False)
    else:
        show_cspa = False
    st.markdown("---")

    HTML_CSPA = HTML_PLOTS["CSPA"][(batch_idx,)]
    html(CSS.replace("min-width: 275px", "min-width: 200px") + HTML_CSPA, height=800)


# s1 = "* Blue means the head **improves** model performance here. Amount of improvement captured by copy-suppression = " + format_word_importances(
#     [f"<b>{s}</b>" for s in ["0%", "20%", "40%", "60%", "80%", "100%"]],
#     [1.0, 0.9, 0.8, 0.7, 0.6, 0.5][::-1],
#     ["" for _ in range(6)],
# ).replace("</mark>", "</mark>, ") + " (i.e. darker colors are what we want to see!)"
# print(s1)




# ! This is where we actually fill in the containers with HTML plots - this depends on whether we are showing the CSPA stuff too

with LOGITS_CONTAINER:
    if is_local and show_cspa:
        HTML_LOGITS_ALL = "".join([CSS, "<h2>Original</h2>", HTML_LOGITS_ORIG, "<h2>Ablated</h2>", HTML_LOGITS_ABLATED, "<h2>CSPA (10.7, direct, frozen)</h2>", HTML_LOGITS_CSPA])
        height = 1400
    else:
        HTML_LOGITS_ALL = "".join([CSS, "<h2>Original</h2>", HTML_LOGITS_ORIG, "<h2>Ablated</h2>", HTML_LOGITS_ABLATED])
        height = 1000
    html(HTML_LOGITS_ALL, height=height)

with DLA_CONTAINER:
    if is_local and show_cspa:
        HTML_DLA_CSPA = HTML_PLOTS["DLA"][(batch_idx, "CSPA", radio_negpos_logits)]
        HTML_DLA_ALL = "".join([CSS, "<h2>Original</h2>", HTML_DLA, "<h2>CSPA (10.7, direct, frozen)</h2>", HTML_DLA_CSPA])
        height = 700
    else:
        HTML_DLA_ALL = CSS + HTML_DLA
        height = 500
    html(HTML_DLA_ALL, height=height)