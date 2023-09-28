#%%
# 
# `This file copies from transformer_lens/rs/callum2/cspa/cspa_implementation.ipynb` but is in .py form, and only focusses on debugging some failures
# 
# The setup and semantic similarity sections must be run. But the rest of the sections can be run independently.

# In[1]:

from transformer_lens.cautils.notebook import *
SEED = 6

if ipython is None:
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-index", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--artifact-base", type=str)

    args = parser.parse_args()
    START_INDEX = args.start_index
    LENGTH = args.length
    ARTIFACT_BASE = args.artifact_base

else:
    ARTIFACT_BASE = "cspa_no_bosses"

ARTIFACT_TO_FORMAT = ARTIFACT_BASE +  "_{seed}_{start_index}_{length}"

t.set_grad_enabled(False)

from transformer_lens.rs.callum2.cspa.cspa_functions import (
    FUNCTION_STR_TOKS,
    concat_dicts,
    get_first_letter,
    begins_with_capital_letter,
    rescale_to_retain_bos,
    get_cspa_results,
    get_cspa_results_batched,
    get_performance_recovered,
    OVProjectionConfig, 
    QKProjectionConfig,
)
from transformer_lens.rs.callum2.utils import (
    parse_str,
    parse_str_toks_for_printing,
    parse_str_tok_for_printing,
    ST_HTML_PATH,
    process_webtext,
)
from transformer_lens.rs.callum2.cspa.cspa_plots import (
    generate_scatter,
    generate_loss_based_scatter,
    show_graphs_and_summary_stats,
    add_cspa_to_streamlit_page,
)
from transformer_lens.rs.callum2.generate_st_html.model_results import (
    get_result_mean,
    get_model_results,
)
from transformer_lens.rs.callum2.generate_st_html.generate_html_funcs import (
    generate_4_html_plots,
    CSS,
)
from transformer_lens.rs.callum2.cspa.cspa_semantic_similarity import (
    get_equivalency_toks,
    get_related_words,
    concat_lists,
    make_list_correct_length,
    create_full_semantic_similarity_dict,
)
from transformer_lens.rs.callum2.utils import get_effective_embedding
from transformers import LlamaForCausalLM, AutoTokenizer
clear_output()

# In[2]:

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

#%%

hf_model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf") # Get LLAMA

#%%

model = HookedTransformer.from_pretrained(
    "llama-7b-hf",
    hf_model=hf_model,
    fold_value_biases=False,
    fold_ln=False,
    tokenizer=tokenizer,
    n_device=1,
    move_to_device=False,
    center_writing_weights=False,
)

# In[3]:

# Discover negative heads!

W_V = model.W_V
W_O = model.W_O

#%%

W_EE = get_effective_embedding(model=model)["W_E (only MLPs)"]

#%%

W_U = model.W_U

#%%

zerred = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

for layer_idx in tqdm(range(model.cfg.n_layers-1, -1, -1)):
    for head_idx in range(model.cfg.n_heads):
        
        cur_W_V = W_V[layer_idx, head_idx] 
        cur_W_O = W_O[layer_idx, head_idx]

        effective_ov_circuit = einops.einsum(
            W_EE,
            cur_W_V,
            cur_W_O, 
            W_U, 
            "v d, d h, h o, o v -> v",
        )

        zerred[layer_idx, head_idx] = effective_ov_circuit.mean()

#%%

px.imshow(
    zerred,
    title="Eigenvalues of W_V @ W_O",
    color_continuous_scale="RdBu",
    zmin=-zerred.abs().max().item(),
    zmax=zerred.abs().max().item(),
    # Label axes
    labels=dict(x="Head", y="Layer", color="Average Logits on Self"),
).show()

#%%

BATCH_SIZE = 500 # 80 for viz
SEQ_LEN = 1000 # 61 for viz

DATA_TOKS, DATA_STR_TOKS_PARSED, indices = process_webtext(seed=SEED, batch_size=(2020 if ipython else START_INDEX+LENGTH), seq_len=SEQ_LEN, model=model, verbose=True, return_indices=True, use_tqdm=True, prepend_bos=True)

#%%

USE_SEMANTICITY = True

if USE_SEMANTICITY:
    from pattern.text.en import conjugate, PRESENT, PAST, FUTURE, SUBJUNCTIVE, INFINITIVE, PROGRESSIVE, PLURAL, SINGULAR
    from nltk.stem import WordNetLemmatizer
    import nltk
    MY_TENSES = [PRESENT, PAST, FUTURE, SUBJUNCTIVE, INFINITIVE, PROGRESSIVE]
    MY_NUMBERS = [PLURAL, SINGULAR]
    from nltk.corpus import wordnet
    nltk.download('wordnet')
    clear_output()

# In[7]:

if USE_SEMANTICITY:
    cspa_semantic_dict = pickle.load(open(ST_HTML_PATH.parent.parent / "cspa/cspa_semantic_dict_full.pkl", "rb"))

else:
    warnings.warn("Not using semanticity unlike old notebook versions!")
    cspa_semantic_dict = {}

# In[8]:

display(HTML("<h2>Related words</h2>This doesn't include tokenization fragments; it's just linguistic."))

for word in ["Berkeley", "pier", "pie", "ring", "device", "robot", "w"]:
    try: print(get_related_words(word, model))
    except: print(get_related_words(word, model)); print("(Worked on second try!)") # maybe because it downloads?

# In[9]:

display(HTML("<h2>Equivalency words</h2>These are the words which will be included in the semantic similarity cluster, during CSPA."))

for tok in [" Berkeley", " Pier", " pier", "pie", " pies", " ring", " device", " robot", "w"]:
    print(f"{tok!r:>10} -> {get_equivalency_toks(tok, model)}")

# In[10]:

if USE_SEMANTICITY:
    table = Table("Source token", "All semantically related", title="Semantic similarity: bidirectional, superstrings, substrings") #  "Top 3 related" in the middle

    str_toks = [" Berkeley", "keley", " University", " Mary", " Pier", " pier", "NY", " ring", " W", " device", " robot", " jump", " driver", " Cairo"]
    print_cutoff = 105 # 70
    def cutoff(s):
        if len(s_str := str(s)) >= print_cutoff: return s_str[:print_cutoff-4] + ' ...'
        else: return s_str

    for str_tok in str_toks:
        top3_sim = "\n".join(list(map(repr, concat_lists(cspa_semantic_dict[str_tok])[:3])))
        bidir, superstr, substr = cspa_semantic_dict[str_tok]
        all_sim = "\n".join([
            cutoff(f"{len(bidir)} bidirectional: {bidir}"),
            cutoff(f"{len(superstr)} super-tokens:  {superstr}"),
            cutoff(f"{len(substr)} sub-tokens:    {substr}"),
        ]) + "\n"
        table.add_row(repr(str_tok), all_sim) # top3_sim in the middle

    rprint(table)
# In[ ]:

# Finally, let's save a mean for later use...

if os.path.exists(os.path.expanduser("~/SERI-MATS-2023-Streamlit-pages/ten_seven_mean.pt")):
    print("Loading from file")
    result_mean = {
        (10, 7): torch.load(os.path.expanduser("~/SERI-MATS-2023-Streamlit-pages/ten_seven_mean.pt")),
        (11, 10): torch.load(os.path.expanduser("~/SERI-MATS-2023-Streamlit-pages/eleven_ten_mean.pt")),
    }

    # torch.save(result_mean[(10, 7)], os.path.expanduser("~/SERI-MATS-2023-Streamlit-pages/ten_seven_mean.pt")) 
    # torch.save(result_mean[(11,10)], os.path.expanduser("~/SERI-MATS-2023-Streamlit-pages/eleven_ten_mean.pt"))

else:
    result_mean = get_result_mean([(10, 7), (11, 10)], DATA_TOKS[-100:, :], model, verbose=True)

#%%