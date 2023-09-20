#%%
# 
# `This file copies from transformer_lens/rs/callum2/cspa/cspa_implementation.ipynb` but is in .py form, and only focusses on debugging some failures
# 
# The setup and semantic similarity sections must be run. But the rest of the sections can be run independently.

# In[1]:

from transformer_lens.cautils.notebook import *
SEED = 6
ARTIFACT_BASE = "cspa_results_q_projection_seed_{SEED}_{start_idx}_{LENGTH}"

if ipython is None:
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-index", type=int)
    parser.add_argument("--length", type=int)

    args = parser.parse_args()
    start_index = args.start_index
    length = args.length
    artifact_name = ARTIFACT_BASE.format(SEED=SEED, start_idx=start_index, LENGTH=length)

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

clear_output()
# In[2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # device="cuda:0",
)
# warnings.warn("Using CPU for CSPA, so it's slow!")
model.set_use_split_qkv_input(False)
model.set_use_attn_result(True)
clear_output()

# In[3]:

BATCH_SIZE = 500 # 80 for viz
SEQ_LEN = 1000 # 61 for viz

current_batch_size = 17 # These are smaller values we use for vizualization since only these appear on streamlit
current_seq_len = 61

NEGATIVE_HEADS = [(10, 7), (11, 10)]
DATA_TOKS, DATA_STR_TOKS_PARSED, indices = process_webtext(seed=SEED, batch_size=(2020 if ipython else start_index+length), seq_len=SEQ_LEN, model=model, verbose=True, return_indices=True, use_tqdm=True, prepend_bos=False)

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
    if ipython is not None:
        Q_PROJECTION_BATCH_START = 0
        Q_PROJECTION_BATCH_END = 20
    Q_PROJECTION_SEQ_LEN = 300

    qk_projection_config = QKProjectionConfig(
        q_direction = "unembedding",
        actually_project=True,
        k_direction = None,
        q_input_multiplier = 2.0,
        query_bias_multiplier = 1.0 if ipython is not None else 0.0, # WARNING - so we can collect data good...
        key_bias_multiplier = 1.0,
        use_same_scaling = False,
        mantain_bos_attention = False,
        model = model,
        save_scores = False,
        swap_model_and_our_max_attention = False,
        swap_model_and_our_max_scores = False,
        capital_adder = 1.25, # 1.25, # 0.75, 0.25, 0.75, # ... so hacky and worth about a percent # 0.25 buys like one percentage point
        save_scaled_resid_pre = False,  
        # save_q_remove_unembed = True,
        # save_query_input_dotter = True,
        # another_direction = extra_direction,
        fix_bos_input=False,
    )

    # ov_projection_config = OVProjectionConfig()
    ov_projection_config = None

    gc.collect()
    t.cuda.empty_cache()
    print("Starting...")
    current_data_toks = DATA_TOKS[Q_PROJECTION_BATCH_START:Q_PROJECTION_BATCH_END, :Q_PROJECTION_SEQ_LEN] if ipython is not None else DATA_TOKS[start_index:start_index+length, :Q_PROJECTION_SEQ_LEN]
    cspa_results_q_projection = get_cspa_results_batched(
        model = model,
        toks = current_data_toks,
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
        computation_device = "cuda",
        do_running_updates = True,
    )
    gc.collect()
    t.cuda.empty_cache()
    cached_cspa = {k:v.detach().cpu() for k,v in cspa_results_q_projection.items()}
    saver_fpath = os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/{ARTIFACT_BASE.format(seed=SEED, length=Q_PROJECTION_BATCH_END-Q_PROJECTION_BATCH_START, start_index=Q_PROJECTION_BATCH_START)}.pt")
    torch.save(cached_cspa, saver_fpath)

else:
    raise Exception("Load in a `cspa_results_q_projection`")

print(  
    "The performance recovered is...",
    get_performance_recovered(cspa_results_q_projection), # ~64
)

# In[31]:

WANDB_PROJECT_NAME = "copy-suppression"

if ipython is None:
    # Initialize wandb
    wandb.init(project=WANDB_PROJECT_NAME)

    # Log as artifact
    artifact = wandb.Artifact(
        name=saver_fpath.split(".")[-2].split("/")[-1],
        type="dataset",
        description="An example tensor"
    )
    artifact.add_file(saver_fpath)
    wandb.log_artifact(artifact)
    wandb.finish()
    time.sleep(10)
    # Delete that file path?
    sys.exit(0)

else:
    # Let's try to load the artifacts!

    # Initialize wandb (optional if already initialized)
    wandb.init(project=WANDB_PROJECT_NAME)
    tensor_datas = []

    for start_index in range(0, 2020, 20):
        artifact_fname = f"cspa_results_q_projection_seed_{SEED}_{start_index}_20"
        try:
            saving_path = Path(os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/artifacts/{artifact_fname}.pt"))

            if os.path.exists(saving_path):
                tensor_datas.append(torch.load(saving_path))
                print("Got", artifact_fname, "from local storage")

            else:
                # Download the artifact
                artifact = wandb.use_artifact(f"{artifact_fname}:latest")
                artifact.download(root=str(saving_path.parent))

                # Load the tensor back into memory
                tensor_datas.append(torch.load(str(saving_path)))

        except Exception as e:
            print("Failed to load", artifact_fname, "because", e)
#%%

all_dicts = {}
for tensor_data in tensor_datas:
    all_dicts = concat_dicts(all_dicts, tensor_data)
torch.save(all_dicts, os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/artifacts/cspa_results_seed_{SEED}_num_batches_{list(all_dicts.values())[0].shape[0]}_try_two.pt")) # Should be around 4GB I think

# Now we need to train!!

#%%

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
    attention_key = "pattern",
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

    cnter = 0
    nocnter = 0

    if verbose:
        for i in range(0, len(indices), 2):
            print(
                "The context was",
                contexts[i],
                end=" ",
            )

            if "all_toks" in locals() or "all_toks" in globals():
                if model.to_single_token(models_words[i+1]) not in all_toks:
                    cnter+=1
                else: 
                    nocnter+=1

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
            true_next_things = next_things[4:] + ["and the model's probability was", str(round(ys[i], 2))] + next_things[:4]+ ["and our probability was", str(round(xs[i+1], 2))]

            print("; ".join(true_next_things))

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

    if "all_toks" in locals() or "all_toks" in globals():
        print("We got", cnter, "cases where the model's word was not in the vocabulary, and", nocnter, "cases where it was")

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
# Freaking 5x as much, too! (The capital_adder can improve stuff by a percentage point or two, but we need more powerful tools)

#%%

# Let's recompute the attention pattern from scores + query bias. And then check that that matches what happens by default
# Then focus on training...

big_fname = os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/artifacts/cspa_results_q_projection_seed_6_20_20.pt")
big_tens = torch.load(big_fname)

#%%

INITIAL_BATCH_SIZE = 1 # So this is as quick as possible for recalculating
initial_scores = big_tens["scores"][0].clone()

print(
    initial_scores.cpu()[:3,:3]
)

key_term = einops.einsum(
    big_tens["scaled_resid_pre"][0],
    model.W_K[10, 7].cpu(),
    "seqK d_model, d_model d_head -> seqK d_head",
) + model.b_K[10, 7].cpu()
query_bias_term = einops.einsum(
    key_term,
    model.b_Q[10, 7].cpu(),
    "seqK d_head, d_head -> seqK",
) / np.sqrt(model.cfg.d_head)

initial_scores[:, 0:] += query_bias_term.unsqueeze(0)[:, 0:] # unsqueeze not necessary strictly! 
new_attention_pattern = initial_scores.softmax(dim=-1) # TODO sanity check that this is the same as what we get with the get_cspa_results function with bias term on!
 
torch.testing.assert_allclose(
    initial_scores.cpu(),
    cspa_results_q_projection["scores"][0].cpu(),
    # big_tens["scores"][0].cpu(),
    # big_tens["normal_pattern"][0].cpu(),
)

#%%

test_data_fnames = {
    start_idx: os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/artifacts/cspa_results_q_projection_seed_6_{start_idx}_20.pt") for start_idx in list(range(60, -20, -20))
}

#%%

W_EE = get_effective_embedding(model)["W_E (only MLPs)"].double()

#%%

# Let's train the gains and biases for all the attention scores!

torch.set_grad_enabled(True)
DIV = 1000
attention_score_bias = torch.nn.Parameter((torch.randn(model.cfg.d_model).double().cuda())/DIV, requires_grad=True)
attention_score_scale = torch.nn.Parameter((torch.randn(model.cfg.d_model).double().cuda())/DIV, requires_grad=True)
opt = torch.optim.AdamW([attention_score_bias, attention_score_scale], lr=1e-4)
NUM_EPOCHS = 1000
all_biases = [attention_score_bias.data.detach().clone()]
all_scales = [attention_score_scale.data.detach().clone()]
LENGTH = 20
TESTING = True

for epoch_idx in range(NUM_EPOCHS):

    with torch.no_grad(): # TODO: we really should factor this out as an `evaluate` function ... please!
        test_loss = torch.tensor(0.0).cuda()
        test_loss_adds = 0
        
        for start_idx, test_data_fname in test_data_fnames.items():
            current_test_data = torch.load(test_data_fname)
            current_test_cuda_data = {k: v.cuda() for k, v in current_test_data.items()}
            current_test_seq_len = current_test_data["scores"].shape[1]

            # Forward pass for test set
            tok_indices_test = DATA_TOKS[start_idx:start_idx+LENGTH, :current_test_seq_len].cuda()
            
            cur_attention_score_scale_test = torch.nn.functional.relu(einops.einsum(
                W_EE[tok_indices_test].detach().clone(),
                attention_score_scale,
                "batch seqK d_model, d_model -> batch seqK",
            ) + 1.0) + 1e-4
            cur_attention_score_bias_test = torch.maximum(einops.einsum(
                W_EE[tok_indices_test].detach().clone(),
                attention_score_bias,
                "batch seqK d_model, d_model -> batch seqK",
            ), torch.tensor(-5.0, dtype=torch.double).cuda())

            initial_attention_score_test = current_test_cuda_data["scores"][:, :current_test_seq_len].clone().double()
            trained_attention_score_test = (torch.maximum(initial_attention_score_test, torch.tensor(-30.0, dtype=torch.double).cuda()) * (cur_attention_score_scale_test.unsqueeze(1))) + cur_attention_score_bias_test.unsqueeze(1)

            key_term_test = einops.einsum(
                current_test_cuda_data["scaled_resid_pre"],
                model.W_K[10, 7],
                "batch seqK d_model, d_model d_head -> batch seqK d_head",
            ) + model.b_K[10, 7]

            query_bias_term_test = einops.einsum(
                key_term_test,
                model.b_Q[10, 7],
                "batch seqK d_head, d_head -> batch seqK",
            ) / np.sqrt(model.cfg.d_head)

            final_attention_score_test = trained_attention_score_test + query_bias_term_test.unsqueeze(1)
            new_attention_pattern_test = final_attention_score_test.softmax(dim=-1)

            test_loss_adds += 1
            test_loss += ((new_attention_pattern_test[:,:,1:] - current_test_cuda_data["normal_pattern"][:, :current_test_seq_len, 1:current_test_seq_len]) ** 2).mean()

        test_loss /= test_loss_adds
    
    if epoch_idx == 0:
        print("Test loss is initially", test_loss.item())

    tot_loss = 0.0
    tot_loss_adds = 0
    for start_idx in (range(80, 700 if TESTING else DATA_TOKS.shape[0], LENGTH)):
        opt.zero_grad()
        loss = torch.tensor(0.0).cuda()
        artifact_fname = ARTIFACT_NAME.format(seed=SEED, start_idx=start_idx, length=LENGTH)
        loading_path = Path(os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/artifacts/{artifact_fname}.pt"))
        current_data = torch.load(str(loading_path))
        current_cuda_data = {k:v.cuda() for k,v in current_data.items()}
        current_seq_len = current_data["scores"].shape[1]

        # Forward pass
        tok_indices = DATA_TOKS[start_idx:start_idx+LENGTH, :current_seq_len].cuda()
        
        cur_attention_score_scale = torch.nn.functional.relu(einops.einsum(
            W_EE[tok_indices].detach().clone(), # ... d_model
            attention_score_scale, # d_model
            "batch seqK d_model, d_model -> batch seqK",
        ) + 1.0) + 1e-4
        cur_attention_score_bias = torch.maximum(einops.einsum(
            W_EE[tok_indices].detach().clone(), # ... d_model
            attention_score_bias, # d_model
            "batch seqK d_model, d_model -> batch seqK",
        ), torch.tensor(-5.0, dtype=torch.double).cuda())

        initial_attention_score = current_cuda_data["scores"][:, :current_seq_len].clone().double()
        trained_attention_score = (torch.maximum(initial_attention_score, torch.tensor(-30.0, dtype=torch.double).cuda()) * (cur_attention_score_scale.unsqueeze(1))) + cur_attention_score_bias.unsqueeze(1)

        key_term = einops.einsum(
            current_cuda_data["scaled_resid_pre"],
            model.W_K[10, 7],
            "batch seqK d_model, d_model d_head -> batch seqK d_head",
        ) + model.b_K[10, 7]
        
        query_bias_term = einops.einsum(
            key_term,
            model.b_Q[10, 7],
            "batch seqK d_head, d_head -> batch seqK",
        ) / np.sqrt(model.cfg.d_head)        

        final_attention_score = trained_attention_score + query_bias_term.unsqueeze(1) # Unsqueeze for seqQ
        new_attention_pattern = final_attention_score.softmax(dim=-1)

        loss = ((new_attention_pattern[:,:,1:] - current_cuda_data["normal_pattern"][:, :current_seq_len, 1:current_seq_len]) ** 2).mean()
        # Backpropagate
        loss.backward()

        tot_loss += loss.item()
        tot_loss_adds += 1

        # Update parameters
        opt.step()

    all_biases.append(attention_score_bias.data.detach().clone())
    all_scales.append(attention_score_scale.data.detach().clone())

    # Print metrics
    print(
        f"Epoch {epoch_idx+1}/{NUM_EPOCHS}",
        f"Train Loss: {tot_loss/tot_loss_adds:.7f}",
        f"Test Loss: {test_loss.item():.7f}",
    )

# %%

# Then actually evaluate this attention pattern on the original task! 

#%%

# See what's up

upwriting = einops.einsum(
    W_EE.cpu(),
    attention_score_bias.cpu(),
    "n_vocab d_model, d_model -> n_vocab",
)
from pprint import pprint
pprint(
    model.to_str_tokens(
        (-upwriting).topk(k=100).indices
    )
)

#%%

# Side project on query bias

W_EE =  model.W_U.clone().T # get_effective_embedding(model)["W_E (only MLPs)"]
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

