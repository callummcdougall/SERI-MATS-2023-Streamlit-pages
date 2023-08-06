## `ioi_and_bos`

This directory is for 2 small investigations:

1. How does the head manage to attend to BOS by default?

Conclusions - when you look at the cosine similarity of "residual stream vector before attn layer 10" and "query bias for head 10.7", it's very positive and in a very tight range for all tokens (between 0.45 and 0.47) whenever position is zero, and the same but very negative for all tokens whenever position isn't zero. So this isn't a function of BOS, it's a function of position. This has implications for how CSPA works; the query-side prediction has to overcome some threshold to actually activate the copy suppression mechanism.

2. What's the perpendicular component of the query, in IOI?

Conclusions - 

* Adding semantically similar tokens `"Mary"`, `"mary"` rather than just `" Mary"` doesn't seem to help.
* Found weak evidence that there's some kind of "indirect prediction", because when you take the perpendicular component and put them through the MLPs it does favour IO over S1 (but the MLPs don't have much impact in IOI so this effect isn't large anyway).

## `st_page`

Hosting all of the Streamlit pages. This isn't for generating any plots (at least I don't use it for that); it's exclusively for hosting pages & storing media files.

The pages are:

1. **OV and QK circuits** - you get to see what tokens are most attended to (QK circuit, prediction-attention) and what tokens are most suppressed (OV circuit). It's a nice way to highlight semantic similarity, and build intuition for how it works.
2. **Browse Examples** - the most important page. You get to investigate OWT examples, and see how all parts of the copy suppression mechanism works. You can:
    * See the loss change per token when you ablate, i.e. find the MIDS (tokens which the head is most helpful for).
    * See the logits pre and post-ablation, as well as the direct logit attribution for this head. *You can confirm that the head is pushing down tokens which appear in context, for most of the MIDS examples.*
    * Look at the attention patterns. *You can confirm that the head is attending to the tokens which it pushes down.*
    * Look at the logit lens before head 10.7. *You can confirm that the head is predicting precisely the words which it is attending to.*

## `cspa`

This is where I get the copy suppression-preserving ablation results. In other words, the stuff that's gonna be in section 3.3 of the paper (and that makes up one of the Streamlit pages).

It also adds to the HTML plots dictionary, for the "Browse Examples" Streamlit page.

## `ov_qk_circuits` 

This generates code for section 3.1, and generates the data for the following Streamlit pages:

1. OV and QK circuits

## `generate_st_html`

This is exclusively for generating the HTML figures that will be on the following Streamlit pages:

2. Browse Examples
3. Test Your Own Examples