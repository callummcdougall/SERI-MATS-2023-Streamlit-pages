# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
for root_dir in [
    os.getcwd().split("rs/")[0] + "rs/callum2/explore_prompts", # For Arthur's branch
    "/app/seri-mats-2023-streamlit-pages/explore_prompts", # For Streamlit page (public)
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/explore_prompts", # For Arthur's branch
    os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0] + "SERI-MATS-2023-Streamlit-pages/explore_prompts", # For Arthur's branch
]:
    if os.path.exists(root_dir):
        break
os.chdir(root_dir)
if root_dir not in sys.path: sys.path.append(root_dir)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
st.set_page_config(layout="wide")
from streamlit.components.v1 import html
from transformer_lens import HookedTransformer
from collections import defaultdict

from streamlit_styling import styling # type: ignore
from generate_html import CSS # type: ignore
from explore_prompts_utils import parse_str_tok_for_printing # type: ignore
from generate_html import generate_4_html_plots # type: ignore

import torch as t
t.set_grad_enabled(False)
import platform

is_local = (platform.processor() != "")
if not is_local:
    st.error(r"""
This page can't be run publicly, because of memory consumption issues. To run the page locally, clone the [GitHub repo](https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages), navigate to `explore_prompts`, run `pip install -r requirements.txt` to install everything in the requirements file, and then run `streamlit run Home.py` to open the page in your browser.
""")
    st.stop()

styling()
# html(CSS)
# st.markdown(CSS, unsafe_allow_html=True)
# from explore_prompts.explore_prompts_backend ModelResults

if "prompt_list" not in st.session_state:
    st.session_state["prompt_list"] = []

NEG_HEADS = ["10.7", "11.10"]
EFFECTS = ["direct", "indirect", "both"]
LN_MODES = ["frozen", "unfrozen"]
ABLATION_MODES = ["mean", "zero"]
ATTENTION_TYPES = ["info-weighted", "standard"]
ATTN_VIS_TYPES = ["large", "small"]

@st.cache_resource(hash_funcs={"tokenizers.Tokenizer": lambda _: None}, show_spinner=False, max_entries=1)
def load_model():
    with st.spinner("Loading model (this only needs to happen once, it usually takes 5-15 seconds) ..."):
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device="cpu"
        ) # .half()
        model.set_use_attn_result(True)
    return model

model = load_model()

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
            cspa=False,
        )
        # Add these new plots to the main dictionary (we have more than 1 prompt active at any one time).
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
    batch_idx = st.sidebar.radio("Pick a sequence", range(BATCH_SIZE), index=BATCH_SIZE-1, format_func=lambda x: st.session_state["prompt_list"][x])
    head_name = st.sidebar.radio("Pick a head", NEG_HEADS) #  + ["both"])
    if head_name == "10.7":
        EFFECTS = ["direct", "indirect", "indirect (excluding 11.10)", "both"]
    effect = st.sidebar.radio("Pick a type of intervention effect", EFFECTS)
    ln_mode = st.sidebar.radio("Pick a type of layernorm mode for the intervention", LN_MODES)
    ablation_mode = st.sidebar.radio("Pick a type of ablation", ABLATION_MODES)
    full_ablation_mode = "+".join([effect, ln_mode, ablation_mode])
    full_ablation_mode_dla = "+".join([ln_mode, ablation_mode])
    # HTML_LOSS = HTML_PLOTS["LOSS"][(batch_idx, head_name, ablation_type)]
    HTML_LOGITS_ORIG = HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)]
    HTML_LOGITS_ABLATED = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, full_ablation_mode)]
    # HTML_DLA = HTML_PLOTS["DLA"][(batch_idx, head_name, "NEG/POS")]
    # HTML_ATTN = HTML_PLOTS["ATTN"][(batch_idx, head_name, vis_name, attn_type)]
    # HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name, remove_self)]
    st.sidebar.markdown(r"""
---

### Explanation for the sidebar options

When we intervene on an attention head, there are 3 choices we can make. We can choose the:

* **Intervention effect** - just the direct path from the head to the final logits, or just the indirect path (everything excluding the direct path), or both.
* **LayerNorm mode** - whether we freeze the LayerNorm parameters to what they were on the clean run, or unfreeze them (so the new residual stream is normalised).
* **Ablation type** - we can mean-ablate, or zero-ablate.
""")

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

        max_loss_color_is_free = st.checkbox("By default, the extreme colors are ± 2.5 cross-entropy loss. Check this box to extremise the colors (so the most important token in this sequence has the most intense color).")

        HTML_LOSS = HTML_PLOTS["LOSS"][(batch_idx, head_name, full_ablation_mode, max_loss_color_is_free)]
        html(CSS.replace("min-width: 275px", "min-width: 130px") + HTML_LOSS, height=400)

    with tabs[1]:

        inner_tabs = st.tabs(["Direct logit attribution of head", "Logprobs before/after ablation"])

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

            HTML_DLA = HTML_PLOTS["DLA"][(batch_idx, head_name, full_ablation_mode_dla, radio_negpos_logits)]
            html(CSS + HTML_DLA, height=400)

        with inner_tabs[1]:
            st.markdown(
r"""
## Logprobs before/after ablation

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

To make the values meaningful, I've scaled them between 0 and 1, so the values you see when hovering can be interpreted as "fraction of the norm of that vector which is in the direction of some particular token's unembedding."

> *Note - this visualisation might be redesigned, because I'm not sure this is the most elegant way to display this information. There's a few hacky things here, e.g. having to subtract the mean unembedding component over source tokens `S` for each destination token `D`, and the fact that `D` will often contain a nontrivial component of its own unembedding because of tied embeddings.*
"""
)
        # remove_self = st.checkbox("Remove self-attention from possible source tokens", value=False)

        HTML_UNEMBEDDINGS = HTML_PLOTS["UNEMBEDDINGS"][(batch_idx, head_name)]
        html(CSS + HTML_UNEMBEDDINGS, height=1500)
