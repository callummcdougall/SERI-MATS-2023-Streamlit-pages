# In[1]:

"""Quad plots but I pivoted away from dot_with_query"""

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


#%%

from transformer_lens.cautils.ioi_dataset import NAMES, PLACES, OBJECTS
raw_tokens = []
for token in NAMES + PLACES + OBJECTS:
    raw_tokens.append(model.to_single_token(" " + token))

# In[9]:

# Prep some bags of words...
# OVERLY LONG because it really helps to have the bags of words the same length

OVERWRITE_WITH_ALL_VOCAB = True
bags_of_words = []
OUTER_LEN = 1
INNER_LEN = model.cfg.d_vocab

t.manual_seed(1243)

all_raw_tokens = []

if OVERWRITE_WITH_ALL_VOCAB:
    assert OUTER_LEN == 1
    assert INNER_LEN == model.cfg.d_vocab
    bags_of_words = [torch.arange(model.cfg.d_vocab)]
    bags_of_words = raw_tokens # ! This was hacked together and so doesn't really represent a bag of words well now
    bags_of_words = torch.randint(0, model.cfg.d_vocab, (2000,))


idx = -1
while idx < 1000: # say
    idx += 1
    cur_tokens = model.tokenizer.encode(dataset[idx])
    cur_bag = []
    
    for i in range(len(cur_tokens)):
        if len(cur_bag) == INNER_LEN:
            break
        if cur_tokens[i] not in all_raw_tokens:
            all_raw_tokens.append(cur_tokens[i])

    bags_of_words = torch.tensor(all_raw_tokens).long()

# In[10]:

embeddings_dict = get_effective_embedding_2(model, use_codys_without_attention_changes=True)

#%%

# Do the OV circuit stuff
embedding = embeddings_dict["W_E (including MLPs)"]

#%%

ten_seven_OV = einops.einsum(
    embedding,
    model.W_V[LAYER_IDX, HEAD_IDX],
    model.W_O[LAYER_IDX, HEAD_IDX],
    "batch d_model, d_model d_head, d_head d_model2 -> batch d_model2",
)

#%%

unembedded = einops.einsum( # Is this too slow ???
    ten_seven_OV.cpu(),
    model.W_U.cpu(),
    "batch d_model2, d_model2 d_vocab -> batch d_vocab",
)

#%%

rankings = (unembedded >= unembedded.diag()).int().sum(dim=-1)

#%%



# In[12]:

# TODO should really make this outer_len and inner_len, but I forgot
# assert all([len(b)==len(bags_of_words[0]) for b in bags_of_words])

#%%
 
better_labels = {
    'W_E (including MLPs)': 'Att_0 + W_E + MLP0', 
    'W_E (no MLPs)': 'W_E',
    'W_E (only MLPs)': 'MLP0',
    'W_U': 'W_U',
}
for embedding_dict_key in embeddings_dict.keys():
    if embedding_dict_key not in better_labels:
        better_labels[embedding_dict_key] = embedding_dict_key

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

USE_QUERY_BIAS = False
USE_KEY_BIAS = False
DO_TWO_DIMENSIONS = False # this means doing things like 2D Attention Matrices

all_log_attentions_to_self = []

for q_side_matrix, k_side_matrix in tqdm(list(itertools.product(embeddings_dict_keys, embeddings_dict_keys))):

    print(f"{q_side_matrix=} {k_side_matrix=}")

    labels.append(f"Q = {better_labels[q_side_matrix]}<br>K = {better_labels[k_side_matrix]}")
    
    try:
        log_attentions_to_self = torch.zeros((len(bags_of_words), len(bags_of_words[0])))
    except:
        log_attentions_to_self = torch.zeros(len(bags_of_words))

    if "K = W_U" in labels[-1]: 
        labels = labels[:-1]
        continue
    if "Q = W_E" in labels[-1]: 
        labels = labels[:-1]
        continue

    results = []

    for outer_idx in tqdm(range(len(bags_of_words))):
        if DO_TWO_DIMENSIONS:
            unnormalized_queries = einops.repeat(
                embeddings_dict[q_side_matrix][bags_of_words[outer_idx]], # [d_model]
                "inner_len d_model -> inner_len another_inner_len d_model",
                another_inner_len=INNER_LEN,
            )
            unnormalized_keys = einops.repeat(
                embeddings_dict[k_side_matrix][bags_of_words[outer_idx]], # [inner_len, d_model],
                "inner_len d_model -> another_inner_len inner_len d_model",
                another_inner_len=INNER_LEN,
            )
        else:
            unnormalized_queries = einops.repeat(
                embeddings_dict[q_side_matrix][bags_of_words[outer_idx]], # [d_model]
                "d_model -> inner_len d_model",
                inner_len=INNER_LEN,
            )
            unnormalized_keys = embeddings_dict[k_side_matrix][torch.arange(model.cfg.d_vocab)] # [inner_len, d_model]

        queryside_vector = embeddings_dict[q_side_matrix][bags_of_words[outer_idx]]
        queryside_normalized = queryside_vector / (queryside_vector.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)
        query = model.W_Q[LAYER, HEAD].T @ queryside_normalized
        if USE_QUERY_BIAS:
            query += model.b_Q[LAYER, HEAD]

        keyside_vectors = embeddings_dict[k_side_matrix][torch.arange(model.cfg.d_vocab)]
        keyside_normalized = keyside_vectors / (keyside_vectors.var(dim=-1, keepdim=True) + model.cfg.eps).pow(0.5)
        key = keyside_normalized @ model.W_K[LAYER, HEAD]
        if USE_KEY_BIAS:
            key += model.b_K[LAYER, HEAD]

        attention_scores = query @ key.T / np.sqrt(model.cfg.d_head)

        assert len(attention_scores.shape) == 1 + int(DO_TWO_DIMENSIONS), attention_scores.shape

        log_attentions_to_self[outer_idx] = (attention_scores >= (attention_scores[bags_of_words[outer_idx]] - 1e-5)).int().sum()

    all_log_attentions_to_self.append(log_attentions_to_self.cpu())
    lines.append(log_attentions_to_self.mean())
    print(lines[-1])

# In[21]:

square_of_values = einops.rearrange(torch.tensor(lines), "(height width) -> height width", height=3)
labels = square_of_values.tolist()

fig = imshow(
    square_of_values.log(),
    # text_auto=True,
    title=f"Average rank of tokens in static QK circuit", #  with {USE_QUERY_BIAS=} {USE_KEY_BIAS=}",
    labels={"x": "Keyside lookup table", "y": "Queryside lookup table", "color": "Average Rank"},
    x = ["W_EE", "W_E", "MLP0"], # x y sorta reversed with imshow
    y = ["W_EE", "W_E", "W_U"],
    color_continuous_midpoint=None,
    range_color=(0, 10), # This manually defines the range of things
    coloraxis=dict(
        colorbar=dict(
        tickvals=np.log(np.array([1, 10, 100, 1000, 10000])),
        ticktext=['1', '10', '100', '1000', '10000']
    )),
    color_continuous_scale="Blues_r",
    return_fig=True,
)

for i, row in enumerate(labels):
    for j, label in enumerate(row):
        fig.add_annotation(
            x=j, # x-coordinate of the annotation
            y=i, # y-coordinate of the annotation
            text=str(round(label, 2)), # text label
            showarrow=False, # don't show an arrow pointing to the annotation
            # color="white",
        )

fig.show()

# %%
