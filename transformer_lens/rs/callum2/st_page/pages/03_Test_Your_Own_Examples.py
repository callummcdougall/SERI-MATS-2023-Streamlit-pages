#%%
# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path

import streamlit as st
st.set_page_config(layout="wide")
from streamlit.components.v1 import html
from contextlib import nullcontext
from transformer_lens import HookedTransformer
from collections import defaultdict
import pickle

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

from transformer_lens.rs.callum2.st_page.streamlit_styling import styling
from transformer_lens.rs.callum2.generate_st_html.model_results import get_model_results
from transformer_lens.rs.callum2.utils import parse_str_tok_for_printing, ST_HTML_PATH
from transformer_lens.rs.callum2.generate_st_html.generate_html_funcs import CSS, generate_4_html_plots
from transformer_lens.rs.callum2.cspa.cspa_functions import get_cspa_results
from transformer_lens.rs.callum2.cspa.cspa_plots import add_cspa_to_streamlit_page

import torch as t
t.set_grad_enabled(False)
import platform

is_local = (platform.processor() != "")
if not is_local:
    st.error(r"""
This page can't be run publicly, because of memory consumption issues. To run the page locally, clone the [GitHub repo](https://github.com/callummcdougall/seri-mats-2023-streamlit-pages), navigate to `explore_prompts`, run `pip install -r requirements.txt` to install everything in the requirements file, and then run `streamlit run Home.py` to open the page in your browser.
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
def load_model_and_semantic_dict_and_result_mean():
    with st.spinner("Loading model (this only needs to happen once, it usually takes 5-15 seconds) ..."):

        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device="cpu"
        ) # .half()
        model.set_use_attn_result(True)

        cspa_semantic_dict = pickle.load(open(ST_HTML_PATH.parent.parent / "cspa/cspa_semantic_dict_full.pkl", "rb"))

        result_mean_as_tensor = t.load(ST_HTML_PATH / "result_mean.pt")
        result_mean = {(10, 7): result_mean_as_tensor[0], (11, 10): result_mean_as_tensor[1]}

    return model, cspa_semantic_dict, result_mean

model, cspa_semantic_dict, result_mean = load_model_and_semantic_dict_and_result_mean()




prompt = st.sidebar.text_area("Prompt", placeholder="Press 'Generate' button to run.", on_change=None)

#%%

def generate_html(
    HTML_PLOTS=None,
    return_html=False,
):

    # Make sure there's some prompt which has been given
    assert prompt is not None
    assert prompt != ""

    context_manager = nullcontext() if return_html else st.spinner("Generating data from prompt...")

    # Spinner while generating data
    with context_manager:

        # Get the number of prompts you've created so far (store it in a list in session state)

        if return_html:
            BATCH_SIZE = 1
        else:
            BATCH_SIZE = len(st.session_state["prompt_list"])
            st.session_state["prompt"] = prompt
            st.session_state["prompt_list"].append(prompt)


        # Get tokens from prompt, and deal with edge case where it's a single string (this shouldn't happen)
        toks = model.to_tokens(prompt)
        str_toks = model.to_str_tokens(toks)
        if isinstance(str_toks[0], str): str_toks = [str_toks]
        # Parse the string tokens for printing
        str_toks_parsed = [list(map(parse_str_tok_for_printing, s)) for s in str_toks]

        # Generate new HTML plots, and CSPA plots
        model_results = get_model_results(
            model,
            toks=toks,
            negative_heads=[(10, 7), (11, 10)],
            result_mean=result_mean,
            verbose=False
        )
        HTML_PLOTS_NEW = generate_4_html_plots(
            model=model,
            data_toks=toks,
            data_str_toks_parsed=str_toks_parsed,
            negative_heads=[(10, 7), (11, 10)],
            model_results=model_results,
            save_files=False,
            result_mean=result_mean,
        )
        cspa_results, s_sstar_pairs, _1, _2, _3 = get_cspa_results(
            model=model,
            toks=toks,
            negative_head=(10, 7), #  this currently doesn't do anything; it's always 10.7
            # components_to_project=["o"],
            interventions = ["ov"],
            K_unembeddings=1.0,
            K_semantic=8,
            semantic_dict=cspa_semantic_dict,
            # effective_embedding="W_E (including MLPs)",
            result_mean=result_mean,
            return_logits=True,
            # use_cuda=False,
            return_dla=True,
        )
        HTML_PLOTS_NEW = add_cspa_to_streamlit_page(
            cspa_results=cspa_results,
            s_sstar_pairs=s_sstar_pairs,
            data_str_toks_parsed=str_toks_parsed,
            model=model,
            HTML_PLOTS=HTML_PLOTS_NEW,
            toks_for_doing_DLA=toks,
            verbose=False,
        )

        # Add these new plots to the main dictionary (we have more than 1 prompt active at any one time).
        HTML_PLOTS = st.session_state.get("HTML_PLOTS", defaultdict(dict))
        for plot_type, dict_of_plots in HTML_PLOTS_NEW.items():
            HTML_PLOTS[plot_type].update({
                (BATCH_SIZE + first_key, *other_keys): html
                for (first_key, *other_keys), html in dict_of_plots.items()
            })
        del HTML_PLOTS_NEW

        if return_html:
            return HTML_PLOTS

        else:
            st.session_state["HTML_PLOTS"] = HTML_PLOTS

if True:
    ht=generate_html(return_html=True)

#%%

# Skip
if False:
    button = st.sidebar.button("Generate", on_click=generate_html)

# Skip these
if False:
    BATCH_SIZE = len(st.session_state["prompt_list"])
    HTML_PLOTS = st.session_state.get("HTML_PLOTS", None)

BATCH_SIZE = 1
HTML_PLOTS = ht

#%%

if BATCH_SIZE > 0:
    # batch_idx = st.sidebar.radio("Pick a sequence", range(BATCH_SIZE), index=BATCH_SIZE-1, format_func=lambda x: (prompt if BATCH_SIZE==1 else st.session_state["prompt_list"][x]))
    batch_idx = 1
    head_name = st.sidebar.radio("Pick a head", NEG_HEADS) #  + ["both"])
    if head_name == "10.7":
        EFFECTS = ["direct", "indirect", "indirect (excluding 11.10)", "both"]
    effect = st.sidebar.radio("Pick a type of intervention effect", EFFECTS)
    ln_mode = st.sidebar.radio("Pick a type of layernorm mode for the intervention", LN_MODES)
    # ablation_mode = st.sidebar.radio("Pick a type of ablation", ABLATION_MODES)
    ablation_mode = "mean"
    full_ablation_mode = "+".join([effect, ln_mode, ablation_mode])
    full_ablation_mode_dla = "+".join([ln_mode, ablation_mode])
    HTML_LOGITS_ORIG = HTML_PLOTS["LOGITS_ORIG"][(batch_idx,)]
    HTML_LOGITS_ABLATED = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, head_name, full_ablation_mode)]
    HTML_LOGITS_CSPA = HTML_PLOTS["LOGITS_ABLATED"][(batch_idx, "CSPA")]
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


st.markdown(
r"""# Test Your Own Examples

This page allows you to input your own prompts, and see how negative heads (10.7 and 11.10) behave on the different tokens in those prompts. It should help build your intuition about how much of the negative heads' behaviour is copy-suppression, and when this is good/bad.

When you enter a prompt in the left-hand sidebar and hit "Generate", that prompt will appear as part of a list below the generate button. You can then navigate between these prompts (in other words you can have multiple prompts active at once, and jump between them).

### A few good example prompts

<details>
<summary>All's fair in love and war.</summary>
<br>

The model will predict `"... love and love"` when you ablate 10.7!

This is an example of 10.7 suppressing **naive copying**. The model has naively predicted that `" love"` will be repeated (further analysis of this example reveals that some earlier attention heads are attending to and positively copying `" love"`), and 10.7 is suppressing this.

</details>

<details>
<summary>I picked up the first box. I picked up the second box. I picked up the third and final box.</summary>
<br>

This is a great example of situations where copy-suppression is good/bad respectively. The model will copy-suppress `" box"` after the tokens `" second"` and `" final"` (which is bad because `" box"` was actually correct here), but it will also heavily copy suppress `" box"` after `" third"`, which is good because `" box"` was incorrect here.

This is an example of 10.7 suppressing a form of **naive induction**. The model completes the pattern in the expected way (thanks to earlier heads), and 10.7 reduces the model's confidence in this prediction.

</details>
""", unsafe_allow_html=True)

if HTML_PLOTS is not None:

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

---
""", unsafe_allow_html=True)

        if is_local:
            show_cspa = st.checkbox("Tick this box to show the CSPA plots for logits & DLA on the second tab, using head 10.7, direct effect, frozen layernorm. *(This is mainly useful during research, so we can see where & why CSPA is failing to capture the effect.)*", value=False)
        else:
            show_cspa = False
        st.markdown("---")

        HTML_CSPA = HTML_PLOTS["CSPA"][(batch_idx,)]
        html(CSS.replace("min-width: 275px", "min-width: 200px") + HTML_CSPA, height=800)





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
# %%
