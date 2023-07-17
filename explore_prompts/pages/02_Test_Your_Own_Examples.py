# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os

try:
    root_dir = os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts"
    os.chdir(root_dir)
except:
    root_dir = "/app/explore_prompts"
    os.chdir(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

import streamlit as st
st.set_page_config(layout="wide")
from streamlit.components.v1 import html
from transformer_lens import HookedTransformer
from collections import defaultdict

from streamlit_styling import styling
from generate_html import CSS
from explore_prompts_utils import parse_str_tok_for_printing
from generate_html import generate_4_html_plots

styling()
# html(CSS)
# st.markdown(CSS, unsafe_allow_html=True)
# from explore_prompts.explore_prompts_backend ModelResults

if "prompt_list" not in st.session_state:
    st.session_state["prompt_list"] = []

NEG_HEADS = ["10.7", "11.10"]
ABLATION_TYPES = ["mean, direct", "zero, direct", "mean, patched", "zero, patched"]
ATTENTION_TYPES = ["info-weighted", "standard"]
ATTN_VIS_TYPES = ["large", "small"]

if "model" not in st.session_state:
    with st.spinner("Loading model (this only needs to happen once) ..."):
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device="cpu"
        )
        model.set_use_attn_result(True)
    
    st.session_state["model"] = model

prompt = st.sidebar.text_area("Prompt", placeholder="Press 'Generate' button to run.", on_change=None)

def generate():
    assert prompt is not None
    assert prompt != ""
    with st.spinner("Generating data from prompt..."):
        BATCH_SIZE = len(st.session_state["prompt_list"])
        st.session_state["prompt"] = prompt
        st.session_state["prompt_list"].append(prompt)
        if "HTML_PLOTS" not in st.session_state:
            st.session_state["HTML_PLOTS"] = defaultdict(dict)
        model: HookedTransformer = st.session_state["model"]
        toks = model.to_tokens(prompt)
        str_toks = model.to_str_tokens(toks)
        if isinstance(str_toks[0], str): str_toks = [str_toks]
        str_toks_parsed = [list(map(parse_str_tok_for_printing, s)) for s in str_toks]
        HTML_PLOTS = generate_4_html_plots(
            model=model,
            data_toks=toks,
            data_str_toks_parsed=str_toks_parsed,
            negative_heads=[(10, 7), (11, 10)],
            save_files=False,
        )
        # Add these new plots to the main dictionary (we have more than 1 prompt active
        # at any one time).
        for k, v in HTML_PLOTS.items():
            v_incremented = {
                (BATCH_SIZE + first_key, *other_keys): html
                for (first_key, *other_keys), html in v.items()
            }
            st.session_state["HTML_PLOTS"][k].update(v_incremented)
        HTML_PLOTS = st.session_state["HTML_PLOTS"]
        

button = st.sidebar.button("Generate", on_click=generate)
# st.sidebar.write(st.session_state["HTML_PLOTS"])

BATCH_SIZE = len(st.session_state["prompt_list"])
HTML_PLOTS = st.session_state.get("HTML_PLOTS", None)

if BATCH_SIZE > 0:
    batch_idx = st.sidebar.radio("Pick a sequence", range(BATCH_SIZE), format_func=lambda x: st.session_state["prompt_list"][x])
    head_name = st.sidebar.radio("Pick a head", NEG_HEADS + ["both"])
    assert head_name != "both", "Both not implemented yet. Please choose either 10.7 or 11.10"
    ablation_type = st.sidebar.radio("Pick a type of ablation", ABLATION_TYPES)
    HTML_LOSS = HTML_PLOTS["LOSS"][(batch_idx, head_name, ablation_type)]
    HTML_LOGITS_ORIG = HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)]
    HTML_LOGITS_ABLATED = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, ablation_type)]
    # HTML_DLA = HTML_PLOTS["DLA"][(batch_idx, head_name, "NEG/POS")]
    # HTML_ATTN = HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, attn_type)]
    # HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name, remove_self)]

st.markdown(
r"""# Browse Examples

This page allows you to input your own prompts, and see how negative heads (10.7 and 11.10) behave on the different tokens in those prompts. It should help build your intuition about how much of the negative heads' behaviour is copy-suppression, and when this is good/bad.

When you enter a prompt in the left-hand sidebar and hit "Generate", that prompt will appear as part of a list below the generate button. You can then navigate between these prompts (in other words you can have multiple prompts active at once, and jump between them).

### A few good example prompts

<details>
<summary>All's fair in love and war.</summary>
<br>

The model will predict `"... love and love"` when you ablate 10.7!

This is an example of 10.7 suppressing **naive copying**.

</details>

<details>
<summary>I picked up the first box. I picked up the second box. I picked up the third and final box.</summary>
<br>

This is a great example of situations where copy-suppression is good/bad respectively. The model will copy-suppress `" box"` after the tokens `" second"` and `" final"` (which is bad because `" box"` was actually correct here), but it will also heavily copy suppress `" box"` after `" third"`, which is good because `" box"` was incorrect here.

This is an example of 10.7 suppressing **naive induction** (specifically, naive fuzzy induction). More generally, it's an example of **breaking the pattern**; reducing the model's overconfidence.

There's also some copy-suppression for `I -> picked`. There isn't copy-suppression for other words in the induction pattern e.g. `picked -> up`, `up -> the`, and `. -> I` are not copy-suppressed, because these are function words.

</details>
""", unsafe_allow_html=True)

if HTML_PLOTS is not None:

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
""", unsafe_allow_html=True)

        html(CSS.replace("min-width: 275px", "min-width: 100px") + HTML_LOSS, height=200)

    with tabs[1]:

        inner_tabs = st.tabs(["Direct logit attribution of head", "Logits before/after ablation"])

        with inner_tabs[0]:

            st.markdown(
r"""
## Direct logit attribution of head

Hover over token T to see what the direct effect of 10.7 is on the logits for T (as a prediction).

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
"""
)
        # remove_self = st.checkbox("Remove self-attention from possible source tokens", value=False)

        HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)]
        html(CSS + HTML_UNEMBEDDINGS, height=1500)
