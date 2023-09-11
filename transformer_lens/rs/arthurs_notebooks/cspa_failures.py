#%%
# 
# `This file copies from transformer_lens/rs/callum2/cspa/cspa_implementation.ipynb` but is in .py form, and only focusses on debugging some failures
# 
# The setup and semantic similarity sections must be run. But the rest of the sections can be run independently.

# In[1]:

from transformer_lens.cautils.notebook import *
t.set_grad_enabled(False)

from transformer_lens.rs.callum2.cspa.cspa_functions import (
    FUNCTION_STR_TOKS,
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

clear_output()

# In[2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cuda",
)
model.set_use_split_qkv_input(False)
model.set_use_attn_result(True)
clear_output()

# In[3]:

BATCH_SIZE = 500 # 80 for viz
SEQ_LEN = 1000 # 61 for viz

current_batch_size = 17 # These are smaller values we use for vizualization since only these appear on streamlit
current_seq_len = 61

NEGATIVE_HEADS = [(10, 7), (11, 10)]
DATA_TOKS, DATA_STR_TOKS_PARSED, indices = process_webtext(seed=6, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, model=model, verbose=True, return_indices=True)

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
result_mean = get_result_mean([(10, 7), (11, 10)], DATA_TOKS[:100, :], model, verbose=True)

# In[30]:

RECALC_CSPA_RESULTS = False

if RECALC_CSPA_RESULTS:
    cspa_results_q_projection = get_cspa_results_batched(
        model = model,
        toks = DATA_TOKS[:Q_PROJECTION_BATCH_SIZE, :Q_PROJECTION_SEQ_LEN],
        max_batch_size = 1,
        negative_head = (10, 7),
        interventions = [],
        qk_projection_config=qk_projection_config,
        ov_projection_config=ov_projection_config,
        K_unembeddings = 1.0,
        K_semantic = 8, # Be very careful making this big... very slow...
        semantic_dict = cspa_semantic_dict,
        result_mean = result_mean,
        use_cuda = True,
        verbose = True,
        compute_s_sstar_dict = False,
        computation_device = "cpu",
    )
    gc.collect()
    t.cuda.empty_cache()
    clear_output()

else:
    # This is just a presaved one of mine...
    cspa_results_q_projection = torch.load(os.path.expanduser("~/SERI-MATS-2023-Streamlit-pages/cspa_results_q_projection_on_cpu.pt"))

# In[31]:

print(
    "The performance recovered is...",
    get_performance_recovered(cspa_results_q_projection), # ~64
)

#%%

JUST_SHOW_STREAMLITS = False

if JUST_SHOW_STREAMLITS:
    PLOT_BATCH_SIZE = 18
    PLOT_SEQ_LEN = 50

else:
    PLOT_BATCH_SIZE = cspa_results_q_projection["kl_div_cspa_to_orig"].shape[0]
    PLOT_SEQ_LEN = cspa_results_q_projection["kl_div_cspa_to_orig"].shape[1]

df = pd.DataFrame({
    "cspa_kl": to_numpy(cspa_results_q_projection["kl_div_cspa_to_orig"][:PLOT_BATCH_SIZE, :PLOT_SEQ_LEN].flatten()),
    "ablated_kl": to_numpy(cspa_results_q_projection["kl_div_ablated_to_orig"][:PLOT_BATCH_SIZE, :PLOT_SEQ_LEN].flatten()),
    # "indices": sum(list(enumerate([[seq_idx for seq_idx in range(Q_PROJECTION_SEQ_LEN)] for _ in range(Q_PROJECTION_BATCH_SIZE)]))),
    "hover_data" : [str((indices[batch_idx], seq_idx, "which is", batch_idx)) for batch_idx in range(PLOT_BATCH_SIZE) for seq_idx in range(PLOT_SEQ_LEN)],
})

fig = px.scatter(
    df,
    x = "cspa_kl",
    y = "ablated_kl",
    hover_data = "hover_data",
)
fig.show()

# Studying some failures of projection, across several different runs. Maybe mean ablation means different things are differently destructive? Scary if so!
# (8, 49), (1, 40), (18, 10), (8, 35)
# (8, 49): Covered below; we pick " homeowners" and " rules" but the model attends to " Aurora" (and " Council"). Note this is on a comma. Maybe we surpress capital letters after that?
# (1, 40): We should attend to private. But instead we attend to prisons. Really weird though, as private is higher predicted, and OV circuit analysis didn't seem to help
# (18, 10): with" -> "TPP" should be suppressed. But " Lee" is predicted more.
# (8, 35): requiring" -> " rentals" is actually attended to. But " homeowners" is predicted more, and hence this ablation picks it more.
# (33, 42): model attends to " remove" -> " Blackberry", we attend to " remove" -> " ban". And Blackberry is the top prediction! Ban is in fact third
# (12, 26): we have half the amount of attention to ' Art' as we should. Context is "\n\n" -> " Art" and the only higher predicted thing is " This" (which is likely a function word)


# In[ ]:

# What are some examples of failures?

def print_attentions(batch_idx, seq_idx):
    print(sorted(list(enumerate(cspa_results_q_projection["pattern"][batch_idx, seq_idx, :seq_idx].tolist())), key=lambda x: x[1], reverse=True))    

# cspa_results_q_projection["pattern"][13, 14, :15] # indices[13] = 35. of -> Neil. We put 90% prob on Neil, but for some reason the model also suppresses "About"
print_attentions(3, 49) # indices[3] = 8. We pick " homeowners" and " rules" but the model attends to " Aurora" and " Council". Note this is on a comma. Maybe we surpress capital letters after that?
# print_attentions(12, 26) # indices[12] = 34. We put too much attention on "This" rather than full 50% on "Art". Failure due to not using semantically similar tokens
# print_attentions(11, 42) # indices[11] = 33. We should attend to Blackberry more. No idea how " meetings" is almost the same amount of attention...
print_attentions(0, 40) # ... [1] we should put weight on " prisons" not private...
print_attentions(5, 10)
print_attentions(3, 35)
print_attentions(11, 42) 
print_attentions(15, 37)
print_attentions(12, 26)

# In[ ]:

# Do a sanity check on some cases where we are doing well, to check that really this "surprising attention" bug is real.
print_attentions(2, 28) 
print("Yeah we absolutely nailed the ~80% attention here")

# Question: what's going wrong? Are our attention scores too high on incorrect, or too low on correct?
# First let's survey the cases where things go well... as a baseline

# In[ ]:

# We would grab the cases where ablated KL >0.1 and CSPA KL <0.05...
# But actually we have way more datapoints, so let's just look at twice as harsh a threshold...

harshness = 2.0

index_relevant = (cspa_results_q_projection["kl_div_ablated_to_orig"] > 0.1 * harshness) &  (cspa_results_q_projection["kl_div_cspa_to_orig"] < 0.05 / harshness)
indices_raw = np.nonzero(to_numpy(index_relevant.flatten()))[0]
success_indices = list(zip(
    indices_raw//index_relevant.shape[-1],
    indices_raw%index_relevant.shape[-1],
    strict=True,
))
print("studying", len(indices), "of", index_relevant.shape[0]*index_relevant.shape[1], "cases")

# In[ ]:

def show_model_cspa_attentions(
    indices,
    cspa_results = cspa_results_q_projection,
    score_key = "pattern",
    show_both_maxes = False,
    verbose=False,
):
    xs=[]
    ys=[]
    cols = []
    models_words = []
    our_words = []
    contexts = []

    for batch_idx, seq_idx in indices:
        current_scores = cspa_results_q_projection[score_key][batch_idx, seq_idx, :seq_idx]
        model_scores = cspa_results_q_projection["normal_"+score_key][batch_idx, seq_idx, :seq_idx]
        assert len(current_scores.shape)==1
        
        max_attention = torch.topk(current_scores, k=1)
        xs.append(max_attention.values.item())
        max_attention_index = max_attention.indices.item()
        ys.append(model_scores[max_attention_index].item())
        # WARNING: we're not passing model nor DATA_TOKS to this function, so handle with care
        our_words.append(model.to_string(DATA_TOKS[batch_idx, max_attention_index].item())) 
        contexts.append("|".join(model.to_str_tokens(DATA_TOKS[batch_idx, max(0,seq_idx-5):seq_idx+1])))
       
        max_model_attention = torch.topk(model_scores, k=1)
        max_model_attention_index = max_model_attention.indices.item()
        models_words.append(model.to_string(DATA_TOKS[batch_idx, max_model_attention_index].item())) 

        if show_both_maxes:
            ys.append(max_model_attention.values.item())
            xs.append(current_scores[max_model_attention_index].item())
            cols.extend(["Our max", "Model's max"])

        else:
            cols.append("BOS" if max_attention_index == 0 else "Not BOS")

    from itertools import chain, repeat
    def interweave(lis):
        """Turn [1,2,3] -> [1,1,2,2,3,3]"""
        return list(chain.from_iterable(repeat(x, 2) for x in lis))
    
    if show_both_maxes:
        indices = interweave(indices)
        our_words = interweave(our_words)
        models_words = interweave(models_words)
        contexts = interweave(contexts)
    
    if verbose:
        for i in range(len(indices)):
            pass
        
            # print(
            #     "The context was",
            #     contexts[i],
            #     "and the model's top word was",
            #     models_words[i],
            #     "probability",
            #     ys[i],
            #     "and our word was",
            #     our_words[i],

            # )

    px.scatter(
        x = xs,
        y = ys,
        color = cols,
        # Label x axis 
        hover_data = {"Indices:": [f"Batch {batch_idx}, seq {seq_idx}" for batch_idx, seq_idx in indices], "Our word:": our_words, "Model word:": models_words, "Context:" : ["|" + context + "|" for context in contexts]},
        labels = {
            "x": "CSPA attention",
            "y": "Model attention",
            "color": "Is max attention on BOS?" if not show_both_maxes else "Which max was taken?",
        },
    ).show()

show_model_cspa_attentions(success_indices)

# Conclusions: in cases where mean ablation was destructuve and projctive CSPA did a reaosnable job, attention in max is somewhat well correlated, and overall there is not muuuch bias; projective just puts slightly more attention (also high BOS attention never happens here)

# %%

# Now let's look at me failures...

fail_index_relevant = (cspa_results_q_projection["kl_div_cspa_to_orig"] > 0.1)
fail_indices_raw = np.nonzero(to_numpy(fail_index_relevant.flatten()))[0]

fail_indices = list(zip(
    fail_indices_raw//fail_index_relevant.shape[-1],
    fail_indices_raw%fail_index_relevant.shape[-1],
    strict=True,
))

print(
    "studying", len(fail_indices_raw), "of", fail_index_relevant.shape[0]*fail_index_relevant.shape[1], "cases",
)

# %%

show_model_cspa_attentions(fail_indices, show_both_maxes=True)

# %%
