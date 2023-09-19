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

clear_output()

# In[2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cpu",
)
warnings.warn("Using CPU for CSPA, so it's slow!")
model.set_use_split_qkv_input(False)
model.set_use_attn_result(True)
clear_output()

# In[3]:

BATCH_SIZE = 500 # 80 for viz
SEQ_LEN = 1000 # 61 for viz

current_batch_size = 17 # These are smaller values we use for vizualization since only these appear on streamlit
current_seq_len = 61
SEED=6

NEGATIVE_HEADS = [(10, 7), (11, 10)]
DATA_TOKS, DATA_STR_TOKS_PARSED, indices = process_webtext(seed=SEED, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, model=model, verbose=True, return_indices=True)

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

# warnings.warn("Making DATA_TOKS be the suffix...")
# DATA_TOKS = DATA_TOKS[-30:]

# In[30]:

extra_direction = torch.load("/root/SERI-MATS-2023-Streamlit-pages/a_new_saved_direction.pt") # A good direction for projection

#%%

RECALC_CSPA_RESULTS = True

if RECALC_CSPA_RESULTS:

    # Empirically, as long as SEQ_LEN large, small BATCH_SIZE gives quite good estimates (experiments about this deleted, too in the weeds)
    Q_PROJECTION_BATCH_SIZE = 20
    Q_PROJECTION_SEQ_LEN = 300

    qk_projection_config = QKProjectionConfig(
        q_direction = "unembedding",
        actually_project=True,
        k_direction = None,
        q_input_multiplier = 2.0,
        query_bias_multiplier = 1.0,
        use_same_scaling = False,
        mantain_bos_attention = False,
        model = model,
        save_scores = True,
        swap_model_and_our_max_attention = False,
        swap_model_and_our_max_scores = False,
        capital_adder = 0.0, # 1.25, # 0.75, 0.25, 0.75, # ... so hacky and worth about a percent # 0.25 buys like one percentage point
        save_scaled_resid_pre = True,    
        # save_q_remove_unembed = True,
        # save_query_input_dotter = True,
        # another_direction = extra_direction,
    )

    # ov_projection_config = OVProjectionConfig()
    ov_projection_config = None

    gc.collect()
    t.cuda.empty_cache()
    print("Starting...")
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
        computation_device = "cuda:0",
        do_running_updates = True,
    )
    gc.collect()
    t.cuda.empty_cache()
    cached_cspa = {k:v.detach().cpu() for k,v in cspa_results_q_projection.items()}
    torch.save(cached_cspa, os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/cspa_results_q_projection_on_cpu_again_seed_{SEED}.pt"))

else:
    # This is just a presaved one of mine...
    cspa_results_q_projection = torch.load(os.path.expanduser("~/SERI-MATS-2023-Streamlit-pages/cspa_results_q_projection_on_cpu_again_seed_{SEED}.pt"))

# In[31]:

print(  
    "The performance recovered is...",
    get_performance_recovered(cspa_results_q_projection), # ~64
)

#%%``

# All of this is seed=6

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

@dataclass
class HiAttentionCounter:
    """Track what the cases where model/CSPA attended to something most were"""
    bos: int = 0
    capital: int = 0
    function_word: int = 0

def show_model_cspa_attentions(
    indices,
    cspa_results = cspa_results_q_projection,
    attention_key = "scores",
    show_both_maxes = False,
    verbose=False,
    do_counting=False,
):
    """Vizualize where the model's and CSPA's attention scores are going
    Also collect some statistics on what the failure cases are. Capital letters? Function words? BOS?"""

    if do_counting:
        assert attention_key == "pattern", "We need to use the pattern for stats because of summing across token positions"

    xs=[]
    ys=[]
    cols = []
    models_words = []
    our_words = []
    contexts = []

    if do_counting:
        model_hi_counter = HiAttentionCounter()
        our_hi_counter = HiAttentionCounter()

    for batch_idx, seq_idx in indices:
        current_attention = cspa_results_q_projection[attention_key][batch_idx, seq_idx, :seq_idx+1]
        model_attention = cspa_results_q_projection["normal_"+attention_key][batch_idx, seq_idx, :seq_idx+1]
        assert len(current_attention.shape)==1
        
        max_attention = torch.topk(current_attention, k=1)
        xs.append(max_attention.values.item())
        max_attention_index = max_attention.indices.item()
        ys.append(model_attention[max_attention_index].item())
        # WARNING: we're not passing model nor DATA_TOKS to this function, so handle with care
        our_words.append(model.to_string(DATA_TOKS[batch_idx, max_attention_index].item())) 
        contexts.append("|".join(model.to_str_tokens(DATA_TOKS[batch_idx, max(0,seq_idx-5):seq_idx+1])))
       
        max_model_attention = torch.topk(model_attention, k=1)
        max_model_attention_index = max_model_attention.indices.item()
        models_words.append(model.to_string(DATA_TOKS[batch_idx, max_model_attention_index].item())) 

        if show_both_maxes:
            ys.append(max_model_attention.values.item())
            xs.append(current_attention[max_model_attention_index].item())
            cols.extend(["Token CSPA had max attention on", "Token that model had max attention on"])
        else:
            cols.append("BOS" if max_attention_index == 0 else "Not BOS")

        # # Process all the counting things...
        if do_counting:
            current_toks = list(set(DATA_TOKS[batch_idx, :seq_idx+1].tolist()))
            model_token_atts = {k: 0.0 for k in current_toks}
            cspa_token_atts = deepcopy(model_token_atts)
            for i in range(seq_idx+1):
                model_token_atts[DATA_TOKS[batch_idx, i].item()] += model_attention[i].item()
                cspa_token_atts[DATA_TOKS[batch_idx, i].item()] += current_attention[i].item()
            model_max_token = max(current_toks, key=lambda x: model_token_atts[x])
            cspa_max_token = max(current_toks, key=lambda x: cspa_token_atts[x])

            # Was the model's token something it attended to more?
            for token, active_attention, other_attention, active_counter in zip(
                [model_max_token, cspa_max_token],
                [model_token_atts, cspa_token_atts],
                [cspa_token_atts, model_token_atts],
                [model_hi_counter, our_hi_counter],
                strict=True,
            ):
                if token == 50256:
                    active_counter.bos += 1
                if active_attention[token] > other_attention[token]:
                    if model.to_single_str_token(token) in FUNCTION_STR_TOKS:
                        active_counter.function_word += 1
                    if begins_with_capital_letter(model.to_single_str_token(token)):
                        active_counter.capital += 1

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
        for i in range(0, len(indices), 2):
            print(
                "The context was",
                contexts[i],
                end=" ",
            )

            next_things = [
                f"and the model's{' top'} word was",
                models_words[i+1].replace("<|endoftext|>", "<|BOS|>"),
                "with probability",
                str(round(ys[i+1], 2)),
                f"and our top word was",
                our_words[i].replace("<|endoftext|>", "<|BOS|>"), 
                "with probability",
                str(round(xs[i], 2)),                
            ]
            next_things = next_things[4:] + ["and the model's probability was", str(round(ys[i], 2))] + next_things[:4]+ ["and our probability was", str(round(xs[i], 2))]

            print("; ".join(next_things))

    px.scatter(
        x = xs,
        y = ys,
        color = cols,
        # Label x axis 
        hover_data = {"Indices:": [f"Batch {batch_idx}, seq {seq_idx}" for batch_idx, seq_idx in indices], "Our word:": our_words, "Model word:": models_words, "Context:" : ["|" + context + "|" for context in contexts]},
        labels = {
            "x": "CSPA attention",
            "y": "Model attention",
            "color": "Is max attention on BOS?" if not show_both_maxes else "What token are we studying here?",
        },
    ).show()

    if do_counting:
        print("Model's hi attention summary:", model_hi_counter, "(and less importantly, our hi attention counter)", our_hi_counter, " and there were a total", len(indices) // (2 if show_both_maxes else 1), "cases")

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

sorted_fail_indices = sorted(fail_indices, key=lambda x: cspa_results_q_projection["kl_div_cspa_to_orig"][x[0], x[1]].item(), reverse=True)

print(
    "We're studying", len(sorted_fail_indices), "of", fail_index_relevant.shape[0]*fail_index_relevant.shape[1], "cases",
)

# %%

show_model_cspa_attentions(sorted_fail_indices, show_both_maxes=True, verbose=True, do_counting=True, attention_key="pattern")

# %%

# Seed 8 was the second batch of things...
# Another observation: we have the max attention on BOS loads more than the model too...

# %%

# Okay so we've identified several issues
# * Underweighting capitals?
# * BOS attention too much?
# * How many function words, too?
# 
# Remember, we have 148 cases
#
# Proportion underweighting capitals: >1/5 seems a big deal. And also needs to be ~4x as big a deal as the inverse
# Function words: >1/15 seems a big deal, needs 3x as many as the inverse.
# BOS too much: >1/4 and needs to be 5x as big as the inverse.
# 
# All of these are plucked out of my ass. The reason that the fractions and factors aren't the same is that some are more interesting than others + smaller subspaces so will be easier to analyse

# %%
# 
# Model's hi attention summary: HiAttentionCounter(bos=6, capital=93, function_word=0) (and less importantly, our hi attention counter) HiAttentionCounter(bos=17, capital=17, function_word=3)  and there were a total 148 cases 
# 
# Wow, we're at >1/2 of the cases being captial cases! The function word does not reproduce, and the BOS thing isn't very siginifcant (~3x as freqent. Still a fiar big though)
#
# Freaking 5x as much, too!
# %%
#
# OK, how do we locate the mystical ` is capital` (presumably!) part of the query resid stream?
# 
# Look at cosine similarities between these secret other directions???
# # Hmm problematic because if it's low we need rethink...
# Is there a "What is the best case scenario experiment to run???" 
# Get all the pairs of (What else other than unembedding shit pointed in the score direction?, How much attention score did it add?s)
#
# Here's another approach: feed in our "failure" sentences into a forward pass, look at diff between capitalised version and not (on keyside?!
# 
# Lol let's find the optimal multiplier to times capital letter words by...
#
#%%

for multiple in [2.5, 2.75, 2.9, 0.0]: 
    batch_size, seq_len, _ = cspa_results_q_projection["scores"].shape
    all_str_tokens = model.to_str_tokens(torch.arange(model.cfg.d_vocab))
    capital_start_tens = torch.tensor(
        [begins_with_capital_letter(x) for x in all_str_tokens]
    )
    toks_capital_letter_start = capital_start_tens[DATA_TOKS[:batch_size, :seq_len]]
    multiplier = toks_capital_letter_start.float() * (multiple - 1.0)
    multiplier += 1.0
    new_scores = cspa_results_q_projection["scores"].clone()
    new_scores *= multiplier.unsqueeze(1) # This should be a property of the key, so is the same across the query (middle) dimension 
    new_bad_bos_pattern = new_scores.softmax(dim=-1)
    new_pattern = rescale_to_retain_bos(
        att_probs = new_bad_bos_pattern, 
        old_bos_probs = cspa_results_q_projection["normal_pattern"][:, :, 0],
    )

    # Calculate distance to the model's probs...

    old_pattern = cspa_results_q_projection["normal_pattern"].clone()

    total_diff = 0

    for batch_idx, seq_idx in sorted_fail_indices:
        old_seq = old_pattern[batch_idx, seq_idx]
        old_max_idx = torch.argmax(old_seq).item()
        total_diff += (old_seq[old_max_idx] - new_pattern[batch_idx, seq_idx, old_max_idx]).abs().item()

    print(multiple, total_diff) # break

# 2.6 is best!

# %%

# Ah failure, performance looks much worse (<50% KL recovered) when we do 2.0 multiplier. Even 1.5x mutliplier seems to be slightly worse than normal (59% recovered [:-(] )

#%%

# New plan: can we learn the important directions (where maybe "capital-ness" is stored?)
#
# After having a think, linear and exact method seems variously flawed, let's train N directions (initially N=1) to minimize L2 of attention probs
#
#%%

t.set_grad_enabled(True)

#%%

def evaluate_directions(
    no_unembed_scaled_resid_pre: Float[torch.Tensor, "batch seqQ ... d_model"], # sometimes we have a K dimension too... also TODO rename from "no_unembed", this just means having removed a key dimension 
    old_scores: Float[torch.Tensor, "batch seqQ seqK"],
    true_pattern: Float[torch.Tensor, "batch seqQ seqK"],
    query_input_dotter: Float[torch.Tensor, "batch seqK d_model"],
    directions: List[Float[torch.Tensor, "d_model"]],    
    do_rescale_to_retain_bos: bool = True,
):
    no_unembed_scaled_resid_pre_shape_prefix = f"batch seqQ{' seqK' if len(no_unembed_scaled_resid_pre.shape)==4 else ''}"

    assert len(directions)==1, "Haven't implemented len(directions)>1 yet"
    direction_projection = einops.einsum(
        no_unembed_scaled_resid_pre,
        directions[0] / directions[0].norm(),
        f"{no_unembed_scaled_resid_pre_shape_prefix} d_model, d_model -> {no_unembed_scaled_resid_pre_shape_prefix}",
    ).unsqueeze(-1) * directions[0]

    extra_attention_score = einops.einsum(
        direction_projection,
        query_input_dotter,
        f"{no_unembed_scaled_resid_pre_shape_prefix} d_model, batch seqK d_model -> batch seqQ seqK",
    ) / np.sqrt(model.cfg.d_head)

    new_attention_score = old_scores + extra_attention_score
    new_attention_pattern = new_attention_score.softmax(dim=-1)

    if do_rescale_to_retain_bos:
        new_attention_pattern = rescale_to_retain_bos(
            att_probs = new_attention_pattern, 
            old_bos_probs = true_pattern[:, :, 0],
        )

    return (new_attention_pattern - true_pattern) ** 2

#%%

TRAIN_SIZE = 75 # because we messed up how we're saving things, TODO make this bigger

assert TRAIN_SIZE <= Q_PROJECTION_BATCH_SIZE

train_unembed_scores = cspa_results_q_projection["scores"][:TRAIN_SIZE]
train_ground_truth = cspa_results_q_projection["normal_pattern"][:TRAIN_SIZE]
train_no_unembed_scaled_resid_pre = cspa_results_q_projection["q_remove_unembed"][:TRAIN_SIZE]
test_unembed_scores = cspa_results_q_projection["scores"][TRAIN_SIZE:]
test_ground_truth = cspa_results_q_projection["normal_pattern"][TRAIN_SIZE:]
test_no_unembed_scaled_resid_pre = cspa_results_q_projection["q_remove_unembed"][TRAIN_SIZE:]

train_query_input_dotter = cspa_results_q_projection["query_input_dotter"][:TRAIN_SIZE]
test_query_input_dotter = cspa_results_q_projection["query_input_dotter"][TRAIN_SIZE:]

for split_name, unembed_scores, ground_truth in zip(
    ["train", "test"], 
    [train_unembed_scores, test_unembed_scores],
    [train_ground_truth, test_ground_truth],
    strict=True,
):
    new_attention = unembed_scores.softmax(dim=-1)
    new_attention = rescale_to_retain_bos(
        att_probs = new_attention, 
        old_bos_probs = ground_truth[:, :, 0],
    )

    print(
        f"The initial {split_name} L2 performance is", 
        round(((new_attention - ground_truth)**2).mean().item(), 5)
    )

# Interesting, values are pretty small

#%%

# Set seed 
torch.manual_seed(41)
direction = torch.nn.Parameter(2.0 * torch.randn(model.cfg.d_model), requires_grad=True)
opt = torch.optim.Adam([direction], lr=0.1)

NUM_EPOCHS = 100

all_directions = [direction.data.detach().clone()]

for epoch_idx in tqdm(range(NUM_EPOCHS)):
    opt.zero_grad() 

    # Forward pass
    all_losses = evaluate_directions(
        no_unembed_scaled_resid_pre = train_no_unembed_scaled_resid_pre,
        old_scores = train_unembed_scores,
        true_pattern = train_ground_truth,
        query_input_dotter = train_query_input_dotter,
        directions = [direction],
        do_rescale_to_retain_bos = False,
    )

    # Calculate loss
    loss = all_losses.mean()
    # loss = all_loss_mean_loss + direction.abs().sum() * 2e-8

    # Backpropagate
    loss.backward()

    # Update parameters
    opt.step()

    all_directions.append(direction.data.detach().clone())

    # Evaluate on test set
    with torch.no_grad():
        all_test_losses = evaluate_directions(
            test_no_unembed_scaled_resid_pre,
            test_unembed_scores,
            test_ground_truth,
            test_query_input_dotter,
            [direction],
            do_rescale_to_retain_bos=False,
        )

        test_loss = all_test_losses.mean()

    # Print metrics
    print(
        f"Epoch {epoch_idx+1}/{NUM_EPOCHS}",
        f"Train Loss: {all_loss_mean_loss.item():.7f}",
        f"Test Loss: {test_loss.item():.7f}",
    )

# %%

# Okay we got 90% cosine similarity on this direction. Does it help
# (A: No, seemed basically irrelevant)
#
#
# %%

# Let's train the biases for all the attention scores

torch.set_grad_enabled(True)
attention_score_bias = torch.nn.Parameter(torch.zeros(model.cfg.d_vocab).double(), requires_grad=True)
attention_score_scale = torch.nn.Parameter(torch.zeros(model.cfg.d_vocab).double(), requires_grad=True)
opt = torch.optim.Adam([attention_score_bias, attention_score_scale], lr=0.007)
NUM_EPOCHS = 1000
all_biases = [attention_score_bias.data.detach().clone()]

for epoch_idx in tqdm(range(NUM_EPOCHS)):

    opt.zero_grad() 

    # Forward pass
    # TODO crucial to remove "pattern" -> "scores"
    tok_indices = einops.repeat(DATA_TOKS[:Q_PROJECTION_BATCH_SIZE, :Q_PROJECTION_SEQ_LEN], "batch seqK -> batch seqQ seqK", seqQ = Q_PROJECTION_SEQ_LEN)
    new_attention_score = (torch.maximum(cspa_results_q_projection["scores"][:, :Q_PROJECTION_SEQ_LEN//2].double(), torch.tensor(-30.0, dtype=torch.double))*(attention_score_scale[tok_indices][:, :Q_PROJECTION_SEQ_LEN//2])+1.0) + attention_score_bias[tok_indices][:, :Q_PROJECTION_SEQ_LEN//2]

    new_attention_pattern = new_attention_score.softmax(dim=-1)
    loss = ((new_attention_pattern - cspa_results_q_projection["normal_pattern"][:Q_PROJECTION_BATCH_SIZE, :Q_PROJECTION_SEQ_LEN//2]) ** 2).mean()

    # Backpropagate
    loss.backward(retain_graph=True)

    # Update parameters
    opt.step()

    all_biases.append(attention_score_bias.data.detach().clone())

    # Evaluate on test set
    with torch.no_grad():
        # new_attention_score = cspa_results_q_projection["scores"][:Q_PROJECTION_BATCH_SIZE, Q_PROJECTION_SEQ_LEN//2:Q_PROJECTION_SEQ_LEN] + attention_score_bias[einops.repeat(DATA_TOKS[:Q_PROJECTION_BATCH_SIZE, :Q_PROJECTION_SEQ_LEN], "batch seqK -> batch seqQ seqK", seqQ = Q_PROJECTION_SEQ_LEN)][:, Q_PROJECTION_SEQ_LEN//2:]
        test_tok_indices = einops.repeat(DATA_TOKS[:Q_PROJECTION_BATCH_SIZE, :Q_PROJECTION_SEQ_LEN], "batch seqK -> batch seqQ seqK", seqQ = Q_PROJECTION_SEQ_LEN)
        new_attention_score = (torch.maximum(cspa_results_q_projection["scores"][:Q_PROJECTION_BATCH_SIZE, Q_PROJECTION_SEQ_LEN//2:Q_PROJECTION_SEQ_LEN].double(), torch.tensor(-30.0, dtype=torch.double)) * attention_score_scale[test_tok_indices][:, Q_PROJECTION_SEQ_LEN//2:] + 1.0) + attention_score_bias[test_tok_indices][:, Q_PROJECTION_SEQ_LEN//2:]

        new_attention_pattern = new_attention_score.softmax(dim=-1)
        test_loss = ((new_attention_pattern - cspa_results_q_projection["normal_pattern"][:, Q_PROJECTION_SEQ_LEN//2:]) ** 2).mean()

    # Print metrics
    print(
        f"Epoch {epoch_idx+1}/{NUM_EPOCHS}",
        f"Train Loss: {loss.item():.7f}",
        f"Test Loss: {test_loss.item():.7f}",
    )
# %%

# Then actually evaluate this attention pattern on the original task! 

#%%

# Side project on query bias

from transformer_lens.rs.callum2.utils import get_effective_embedding
W_EE = get_effective_embedding(model)["W_E (only MLPs)"]
b_Q = model.b_Q[10, 7]
W_K = model.W_K[10, 7]
query_bias_words = (+einops.einsum(
    W_EE,
    W_K,
    b_Q,
    "n_vocab d_model, d_model d_head, d_head -> n_vocab",
)).topk(k=50)

from pprint import pprint

pprint(
    list(
        zip(
            model.to_str_tokens(query_bias_words.indices)
            , query_bias_words.values.tolist()
        )
    )
)

# %%

