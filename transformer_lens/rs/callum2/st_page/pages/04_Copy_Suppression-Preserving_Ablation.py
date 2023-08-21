# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
st.set_page_config(layout="wide")
import pickle
import gzip

import platform
is_local = (platform.processor() != "")

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

from transformer_lens.rs.callum2.st_page.streamlit_styling import styling
from transformer_lens.rs.callum2.utils import ST_HTML_PATH
from transformer_lens.rs.callum2.cspa.cspa_plots import (
    generate_scatter,
    generate_loss_based_scatter,
)

import torch as t
t.set_grad_enabled(False)

styling()

st.markdown(
r"""
# Copy Suppression Classification

Hopefully the previous page built up your intuitions for copy suppression. Here, we'll get a bit more quantitative, and define an ablation procedure which deletes all signal except the exact copy suppression mechanism that we've defined. If our understanding of copy suppression is correct, then we should expect this ablation to not significantly change the model's performance, particularly in situations where it has a large effect on the model's loss.

As a shorthand, we'll call this ablation **copy suppression-preserving ablation** (CSPA), because it's designed to delete all signal **except** that which we understand to be copy-suppression.

### Details of ablation
""")
with st.expander("Click here for the details of how CSPA works."):
    st.markdown(
r"""
First, note that we can write the output of an attention head at destination position $d$ as:

$$
\sum_{s=1}^{d} A_{ds} v_s^T W_O
$$

where $v_s$ is the value vector at source position $S$, $A_{ds}$ is the attention from $d$ to $S$, and $W_O$ is the output matrix of the attention head.

I'll use lowercase letters $s$ and $d$ to refer to the sequence positions of the tokens, and uppercase $S$ and $D$ to refer to the token identities.

#### How do we ablate the OV circuit?

We believe that the main function of the matrix $W_{OV}$ (when applied to source token $S$) is to map the embedding of $S$ to the negative of the unembedding of $S$. So, a natural thing to do would be to project each vector $A_{ds} v_s^T W_O$ onto the unembedding for $S$. However, this misses a small nuance (which we discussed in the "OV and QK circuits" section) - **heads like 10.7 suppress all tokens in a similarity cluster to $S$.** For example, if it attends back to the token `' pier'`, it will suppress both `' Pier'` and `' pier'`, in a way which isn't fully captured just by the fact that these two unembedding vectors have high cosine similarity. 

So, rather than projecting this vector onto just the unembdding of $S$, we project it onto the span of the $k_s$ unembedding vectors for tokens $\{S'_1, ..., S'_{k_s}\}$ which are most negatively copied by the OV circuit (we've used $k_s=5$ for the results below).

This is obviously a trade-off: if $k_u$ is small then we'd only capture pure copy-suppression, but if $k_u$ is too large then we might get information leakage, where we capture more than just the mechanism which we believe copy-suppression is using. To convince yourself that $k_u=5$ is an appropriate choice, you should go to the "OV and QK circuits" page, and see what kind of tokens this gives us. You should usually find that the top 5 tokens are semantically related in a very obvious way.

#### How do we ablate the QK circuit? 

Our theory is that destination token $D$ only attends back to source token $S$ if we are predicting a word at $D$ which is semantically similar to $S$ (call it $S'$). To filter out everything which doesn't fit this pattern, we can do the following:

* Take all the pairs $(S, S')$ of source tokens and tokens which they suppress (which we got from the OV circuit ablation step),
* Pick the top $k_u$ of these pairs, sorted by the size of the unembedding of $S'$ in the residual stream at destination token $D$ (we used $k_u = 10$ for the results below),
* Apply the same projection as we described before at each source token $S$, but only using the $S'$ terms in the top $k_u$ pairs.

So the final expression for the thing we move to destination token $D$ is:

$$
\sum_{s=1}^{d} A_{ds} v_s^T W_O P_{\{u_{s'}: (s, s') \in K_U\}}
$$

where P_X is the matrix which projects onto the subspace spanned by the vectors in $X$, and in this case $X = {\{u_{s'}: (s, s') \in K_U\}}$ is the set of all unembedding vectors s.t. 

#### What exact form of ablation do we use? 

We used direct ablation, i.e. just intervening at the final value of the residual stream and replacing our head's output with its new output. This is because we think that this captures the vast majority of the model's impact. Note that there's some subtletly here - head 11.10 is also doing copy-suppression, but if 10.7 has already suppressed a token then maybe 11.10 won't need to suppress it any more - in other words, 11.10 backs up 10.7's copy-suppression behaviour. If we measure the indirect effect of 10.7 *through all components except head 11.10*, we find that the effect is very small.

#### Mean-ablation vs zero-ablation

The expression above is what we'd get if we were doing zero-ablation (because projecting is equivalent to setting all other components to zero). Instead, we use mean ablation, which means we subtract the mean of each term $A_{ds} v_s^T W_O$ before doing the projection (where the mean is taken across all sequences, and indices $s$ and $d$). Then we add the mean back after projecting.

#### What percentage of components are ablated?

Each term from the sum above is projected onto a matrix $P_{\{u_{s'}: (s, s') \in K_U\}}$, and the sum of the ranks of all of these matrices (over $s$) is the size of the set $K_U$. So the total number of dimensions we're preserving in this operation is $K_U$ (in our results below, this is 10). The total number of available dimensions the model has for this operation is $d_{model} * L$, where $d_{model} = 768$ is the dimension of the model's residual stream, and $L$ is the sequence length. So the percentage of dimensions we're preserving is $\frac{K_U}{d_{model} * L}$. In our results below, we used a sequence length $L=70$, so the average sequence length per token we performed this ablation on was $70/2=35$. This gives us a percentage of $\frac{10}{768 * 35} = 0.04\%$.""")

st.markdown(r"""
### What does this ablation look like in practice?

""", unsafe_allow_html=True)

with st.expander("Click on this dropdown for a step-by-step walkthrough of an example where CSPA captures almost all of the effect."):
    st.markdown(
r"""
Take sequence #32 as an example (you can find it on the "Browse Examples") page on the left-hand sidebar. At the text `' University of California'`, the `' of'` token attends back to the `'keley' token at the start of the sequence (which is part of the tokenization of the word "Berkeley" with no space prepended). The diagram below shows how copy-suppression happens here.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/example01.png" width="750">

How does CSPA preserve this?

First, we look at all tokens $S$ in context, and each of their 5 closest semantic neighbours $S'$. One such pair $(S, S')$ is `('keley', ' Berkeley')`, as we can see from the OV circuit results below:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/example02.png" width="300">

Next, we look at the **logit lens** for the `' of'` token. We can see that `' Berkeley'` is predicted with high probability:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/example03.png" width="500">

Note that it's not the highest predicted token (it has rank 24). But this is much larger than all other $S'$ tokens for the other pairs $(S, S')$ we found, and that's enough to make sure that token $S$ is attended to rather than anything else in context.

Finally, when we apply our ablation, we'll be preserving the component of the term moved from the `'keley'` source token to the `' of'` destination token, in the direction of the unembedding for `' Berkeley'`. Since suppressing the `' Berkeley'` prediction is the main way in which this head affects the loss at this particular token, we can see how CSPA will preserve this effect.
""", unsafe_allow_html=True)

st.markdown(
r"""
### What percentage of 10.7's behaviour is copy-suppression?
""")

@st.cache_data(show_spinner=False, max_entries=1)
def load_fig_dict():
    with gzip.open(ST_HTML_PATH / "CSPA_figs.pkl", "rb") as f:
        fig_dict = pickle.load(f)
    return fig_dict

fig_dict = load_fig_dict()

with st.expander("Click on this dropdown for an analysis of what percentage of the head's behaviour is is preserved by CSPA."):
    st.markdown(
r"""
We measure the following two things:

* How much the loss increases when you entirely ablate this head,
* How much the loss increases when you perform CSPA on this head.

If this head is doing copy-suppression in the way we believe it to be, then we'd expect CSPA to not significantly increase the loss (because you're preserving the main mechanism via which this head is affecting the loss). We ran, this experiment, and you can see the results below.

""")

    st.markdown(
r"""
#### Scatter plot

The first plot shows a scatter of the two different changes in loss (i.e. the x-axis is the increase in loss from mean ablation, and the y-axis is the increase in loss from CSPA). 

The key observations here are:

1. **CSPA explains most of how the head affects the loss, either positively or negatively**. The standard deviation is much smaller on the y-axis than on the x-axis.
2. **CSPA explains most of how the head improves the model's performance**. The mean is much smaller on the y-axis.
3. **CSPA is especially good at explaining the head's effect when it has a large effect on the loss**. There are 214 points where ablating the head changes cross-entropy loss by more than 0.5, but there are only 8 points where performing CSPA does.

You can hover over each point to see the sequence & token it corresponds to, and then look at the corresponding sequence & token in the "Browse Examples" page. For some of the examples where CSPA fails to capture the head's effect, it still seems like copy-suppression, but attention is distributed over several source tokens so it's not surprising that limiting to $K_u = 10$ misses a lot of the signal. However, there are some examples which genuinely don't look like copy-suppression. Also, we still have some false positives (situations where you might expect copy-suppression to be happening but it doesn't), and we won't have fully understood the heads' behaviour unless we understand these too.
""")

    checkbox = st.checkbox("Highlight red all the points from the 'Browse Examples' page (so you can cross-reference them)", value=False)

    if checkbox:
        k = 200 if is_local else 51
    else:
        k = "samecolor"

    st.plotly_chart(
        fig_dict[k],
        use_container_width = True,
    )

    st.markdown(
r"""
#### Line plot

The next plot shows similar information, but we've simplified the plot by bucketing the points according to how much they affect the loss, and averaging over both the x and y values. This more crisply shows the pattern we can see in the plot above: CSPA explains most of the head's effect, and especially when it has a large effect on the loss.
""")

    st.plotly_chart(
        fig_dict["loss-based scatter"],
        use_container_width = False,
    )