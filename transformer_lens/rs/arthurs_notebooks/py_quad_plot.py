# In[1]:

from transformer_lens.cautils.notebook import *
from transformer_lens.cautils.utils import lock_attn
from transformer_lens.rs.callum.keys_fixed import get_effective_embedding_2
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import dot_with_query

# In[2]:

MODEL_NAME = "gpt2-small"
# MODEL_NAME = "gpt2-large"
# MODEL_NAME = "solu-10l"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)

model.set_use_attn_result(False)
model.set_use_split_qkv_input(True)

clear_output()


# In[3]:


LAYER_IDX, HEAD_IDX = {
    "SoLU_10L1280W_C4_Code": (9, 18), # (9, 18) is somewhat cheaty
    "gpt2": (10, 7),
}[model.cfg.model_name]


warnings.warn("Changing to duplicate token")
LAYER_IDX, HEAD_IDX = 3, 0

W_U = model.W_U
W_Q_negative = model.W_Q[LAYER_IDX, HEAD_IDX]
W_K_negative = model.W_K[LAYER_IDX, HEAD_IDX]

W_E = model.W_E

# ! question - what's the approximation of GPT2-small's embedding?
# lock attn to 1 at current position
# lock attn to average
# don't include attention


# In[4]:


full_QK_circuit = FactoredMatrix(W_U.T @ W_Q_negative, W_K_negative.T @ W_E.T)

indices = t.randint(0, model.cfg.d_vocab, (250,))
full_QK_circuit_sample = full_QK_circuit.A[indices, :] @ full_QK_circuit.B[:, indices]

full_QK_circuit_sample_centered = full_QK_circuit_sample - full_QK_circuit_sample.mean(dim=1, keepdim=True)

imshow(
    full_QK_circuit_sample_centered,
    labels={"x": "Source / key token (embedding)", "y": "Destination / query token (unembedding)"},
    title="Full QK circuit for negative name mover head",
    width=700,
)


# In[5]:


raw_dataset = load_dataset("stas/openwebtext-10k")
train_dataset = raw_dataset["train"]
dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]


# In[6]:

# def fwd_pass_lock_attn


# for i, s in enumerate(dataset):
#     loss_hooked = fwd_pass_lock_attn0_to_self(model, s)
#     print(f"Loss with attn locked to self: {loss_hooked:.2f}")
#     loss_hooked_0 = fwd_pass_lock_attn0_to_self(model, s, ablate=True)
#     print(f"Loss with attn locked to zero: {loss_hooked_0:.2f}")
#     loss_orig = model(s, return_type="loss")
#     print(f"Loss with attn free: {loss_orig:.2f}\n")

#     # gc.collect()

#     if i == 5:
#         break


# In[7]:


if "gpt" in model.cfg.model_name: # sigh, tied embeddings
    # sanity check this is the same 

    def remove_pos_embed(z, hook):
        return 0.0 * z

    # setup a forward pass that 
    model.reset_hooks()
    model.add_hook(
        name="hook_pos_embed",
        hook=remove_pos_embed,
        level=1, # ???
    ) 
    model.add_hook(
        name=utils.get_act_name("pattern", 0),
        hook=lock_attn,
    )
    logits, cache = model.run_with_cache(
        torch.arange(1000).to(device).unsqueeze(0),
        names_filter=lambda name: name=="blocks.1.hook_resid_pre",
        return_type="logits",
    )


    W_EE_test = cache["blocks.1.hook_resid_pre"].squeeze(0)
    W_EE_prefix = W_EE_test[:1000]

    assert torch.allclose(
        W_EE_prefix,
        W_EE_test,
        atol=1e-4,
        rtol=1e-4,
    )


# In[8]:


NAME_MOVERS = {
    "gpt2": [(9, 9), (10, 0), (9, 6)],
    "SoLU_10L1280W_C4_Code": [(7, 12), (5, 4), (8, 3)],
}[model.cfg.model_name]

NEGATIVE_NAME_MOVERS = {
    "gpt2": [(LAYER_IDX, HEAD_IDX), (11,10)],
    "SoLU_10L1280W_C4_Code": [(LAYER_IDX, HEAD_IDX), (9, 15)], # second one on this one IOI prompt only... 
}[model.cfg.model_name]


# In[9]:


# Prep some bags of words...
# OVERLY LONG because it really helps to have the bags of words the same length

bags_of_words = []

OUTER_LEN = 50
INNER_LEN = 100

idx = -1
while len(bags_of_words) < OUTER_LEN:
    idx += 1
    cur_tokens = model.tokenizer.encode(dataset[idx])
    cur_bag = []
    
    for i in range(len(cur_tokens)):
        if len(cur_bag) == INNER_LEN:
            break
        if cur_tokens[i] not in cur_bag:
            cur_bag.append(cur_tokens[i])

    if len(cur_bag) == INNER_LEN:
        bags_of_words.append(cur_bag)


# In[10]:


embeddings_dict = get_effective_embedding_2(model)
# TODO append some more embeddings to this dict to see how well they do

# In[11]:


# embeddings_dict_keys

better_labels = {
    'W_E (including MLPs)': 'Att_0 + W_E + MLP0', 
    'W_E (no MLPs)': 'W_E',
    'W_E (only MLPs)': 'MLP0',
    'W_U': 'W_U',
}

# In[12]:

# TODO should really make this outer_len and inner_len, but I forgot
assert all([len(b)==len(bags_of_words[0]) for b in bags_of_words])

# In[18]:

# Getting just diag patterns for a single head

from transformer_lens import FactoredMatrix

LAYER = 10
HEAD = 7
NORM = True

all_results = []
embeddings_dict_keys = sorted(embeddings_dict.keys())
labels = []

data = []
lines = []

USE_QUERY_BIAS = True
USE_KEY_BIAS = True

for q_side_matrix, k_side_matrix in tqdm(list(itertools.product(embeddings_dict_keys, embeddings_dict_keys))):

    print(f"{q_side_matrix=} {k_side_matrix=}")

    labels.append(f"Q = {better_labels[q_side_matrix]}<br>K = {better_labels[k_side_matrix]}")
    
    log_attentions_to_self = torch.zeros((len(bags_of_words), len(bags_of_words[0])))

    if "K = W_U" in labels[-1]: 
        labels = labels[:-1]
        continue
    if "Q = W_E" in labels[-1]: 
        labels = labels[:-1]
        continue

    results = []

    for outer_idx in tqdm(list(range(OUTER_LEN))):
        unnormalized_queries = einops.repeat(
            embeddings_dict[q_side_matrix][bags_of_words[outer_idx]],
            "inner_len d_model -> inner_len another_inner_len d_model",
            another_inner_len=INNER_LEN,
        )
        unnormalized_keys = einops.repeat(
            embeddings_dict[k_side_matrix][bags_of_words[outer_idx]], # [inner_len, d_model],
            "inner_len d_model -> another_inner_len inner_len d_model",
            another_inner_len=INNER_LEN,
        )
        attn_scores = dot_with_query(
            unnormalized_keys = unnormalized_keys,
            unnormalized_queries = unnormalized_queries,
            model = model,
            layer_idx = LAYER,
            head_idx = HEAD,
            add_key_bias = USE_KEY_BIAS,
            add_query_bias = USE_QUERY_BIAS,
            # normalize_keys = True, # True by default            
        )
        assert len(attn_scores.shape) == 2, attn_scores
        # TODO sort out the fact we have inner len and outer len now...

        attn = attn_scores.softmax(dim=-1)
        log_attentions_to_self[outer_idx] = attn[torch.arange(INNER_LEN), torch.arange(INNER_LEN)]

    lines.append(log_attentions_to_self.mean())
    print(lines[-1])

# In[21]:

imshow(
    einops.rearrange(torch.tensor(lines), "(height width) -> height width", height=3),
    text_auto=True,
    title="Average token self-attention in static QK circuit",
    labels={"x": "Keyside lookup table", "y": "Queryside lookup table", "color": "Self-Attention"},
    x = ["W_EE", "W_E", "MLP0"], # x y sorta reversed with imshow
    y = ["W_EE", "W_E", "W_U"],
)

# In[23]:

model.W_U.norm()

# In[ ]:





# In[17]:


imshow(
    all_results,
    facet_col=0,
    facet_col_wrap=len(embeddings_dict)-1,
    facet_labels=labels,
    title=f"Sample of average log softmax for attention approximations with different effective embeddings: head {LAYER}.{HEAD}",
    labels={"x": "Key", "y": "Query"},
    height=900, width=900
)


# In[ ]:


scores = t.zeros(12, 12).float().to(device)

for layer, head in tqdm(list(itertools.product(range(12), range(12)))):
    results = []
    for idx in range(OUTER_LEN):
        softmaxed_attn = get_EE_QK_circuit(
            layer,
            head,
            model,
            show_plot=False,
            random_seeds=None,
            bags_of_words=bags_of_words[idx:idx+1],
            mean_version=False,
            W_E_query_side=embeddings_dict["W_U (or W_E, no MLPs)"],
            W_E_key_side=embeddings_dict["W_E (including MLPs)"],  # "W_E (only MLPs)"
        )

        # now sort each 

        results.append(softmaxed_attn.diag().mean())

    results = sum(results) / len(results)

    scores[layer, head] = results

imshow(scores, width=750, labels={"x": "Head", "y": "Layer"}, title="Prediction-attn scores for bag of words (including MLPs in embedding)")


# In[ ]:


scores = t.zeros(12, 12).float().to(device)

for layer, head in tqdm(list(itertools.product(range(12), range(12)))):
    results = []
    for idx in range(OUTER_LEN):
        softmaxed_attn = get_EE_QK_circuit(
            layer,
            head,
            model,
            show_plot=False,
            random_seeds=None,
            bags_of_words=bags_of_words[idx:idx+1],
            mean_version=False,
            W_E_query_side=embeddings_dict["W_U (or W_E, no MLPs)"],
            W_E_key_side=embeddings_dict["W_E (only MLPs)"],  # 
        )
        results.append(softmaxed_attn.diag().mean())

    results = sum(results) / len(results)

    scores[layer, head] = results

imshow(scores, width=750, labels={"x": "Head", "y": "Layer"}, title="Prediction-attn scores for bag of words (only MLPs in embedding)")


# In[ ]:


print("Do a thing where we make the softmax denominator the same???")


# In[ ]:





# In[ ]:




