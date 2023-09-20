#%%
# 
# `This file copies from transformer_lens/rs/callum2/cspa/cspa_implementation.ipynb` but is in .py form, and only focusses on debugging some failures
# 
# The setup and semantic similarity sections must be run. But the rest of the sections can be run independently.

# In[1]:

from transformer_lens.cautils.notebook import *

if ipython is None:
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-index", type=int)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    start_index = args.start_index
    length = args.length

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

clear_output()
# In[2]:

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cuda:0",
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
SEED=6

NEGATIVE_HEADS = [(10, 7), (11, 10)]
DATA_TOKS, DATA_STR_TOKS_PARSED, indices = process_webtext(seed=SEED, batch_size=(2020 if ipython else start_index+length), seq_len=SEQ_LEN, model=model, verbose=True, return_indices=True, use_tqdm=True)

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
        Q_PROJECTION_BATCH_SIZE = 100
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
    current_data_toks = DATA_TOKS[Q_PROJECTION_BATCH_SIZE:Q_PROJECTION_BATCH_SIZE*2, :Q_PROJECTION_SEQ_LEN] if ipython is not None else DATA_TOKS[start_index:start_index+length, :Q_PROJECTION_SEQ_LEN]
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
        computation_device = "cuda:0",
        do_running_updates = True,
    )
    gc.collect()
    t.cuda.empty_cache()
    cached_cspa = {k:v.detach().cpu() for k,v in cspa_results_q_projection.items()}
    saver_fpath = os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/cspa_results_q_projection_seed_{SEED}_{'null' if ipython is not None else start_index}_{'null' if ipython is not None else length}.pt")
    torch.save(cached_cspa, saver_fpath)

else:
    # This is just a presaved one of mine...
    cspa_results_q_projection = torch.load(os.path.expanduser(f"~/SERI-MATS-2023-Streamlit-pages/cspa_results_q_projection_on_cpu_again_seed_{SEED}.pt"))

#%%

warnings.warn("We factored this out of cspa_failures.py and it wasn't promising, so unlikely to be helpful : (")
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
# (A: No, seemed basically irrelevant. Could be bugged, but I'm not very excited about this direction as it seems things kinda depend on what the key token is?)
