# In[1]:

"""Quad plots but I pivoted away from dot_with_query"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch 
# One device assertion
assert torch.cuda.device_count() == 1

from transformer_lens.cautils.notebook import *
from transformer_lens.cautils.utils import lock_attn
from transformer_lens.rs.callum2.utils import get_effective_embedding
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

# LAYER_IDX, HEAD_IDX = 9, 6
# warnings.warn("Using wrong thing")

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

# In[10]:

embeddings_dict = get_effective_embedding(model, use_codys_without_attention_changes=True)

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

unembedded = einops.einsum( # Is this too slow ??? Not really. Takes 10 seconds but works. I think it will
    ten_seven_OV,
    model.W_U,
    "batch d_model2, d_model2 d_vocab -> batch d_vocab",
)

#%%

unembedded = unembedded.to("cpu")
gc.collect()
torch.cuda.empty_cache()
rankings = (unembedded <= unembedded.diag()[:, None]).int().sum(dim=-1) # ie repressed even harder!

#%%

ranking_top_10 = (rankings<=10)
print(ranking_top_10.int().sum())
print(ranking_top_10.int().sum()/model.cfg.d_vocab)
bags_of_words = ranking_top_10.int().nonzero()[:, 0]
bags_of_words = torch.arange(model.cfg.d_vocab)

#%%

worst = rankings.argsort(
    descending=True,
)

#%%

failures = model.to_str_tokens(
    worst[:-46533],
)

#%%

# np.random.shuffle(failures)

#%%

print(failures[:100])

# In[12]:

# TODO should really make this outer_len and inner_len, but I forgot
# assert all([len(b)==len(bags_of_words[0]) for b in bags_of_words])

#%%
 
better_labels = {
    'W_E (including MLPs)': 'W_EE', 
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

all_results = []
labels = []
data = []
lines = []

USE_QUERY_BIAS = False
USE_KEY_BIAS = False
DO_TWO_DIMENSIONS = False # this means doing things like 2D Attention Matrices

all_log_attentions_to_self = []

b_K = model.b_K[LAYER_IDX, HEAD_IDX]
b_Q = model.b_Q[LAYER_IDX, HEAD_IDX]
W_Q = model.W_Q[LAYER_IDX, HEAD_IDX]
W_K = model.W_K[LAYER_IDX, HEAD_IDX]
eps = model.cfg.eps
d_vocab = model.cfg.d_vocab
d_head = model.cfg.d_head
del model 
gc.collect()
t.cuda.empty_cache()

#%%

for q_side_matrix, k_side_matrix in tqdm(list(itertools.product(embeddings_dict.keys(), embeddings_dict.keys()))):

    print(f"{q_side_matrix=} {k_side_matrix=}")

    labels.append(f"Q = {better_labels[q_side_matrix]}<br>K = {better_labels[k_side_matrix]}")
    
    try:
        log_attentions_to_self = torch.zeros((len(bags_of_words), len(bags_of_words[0])))
    except:
        log_attentions_to_self = torch.zeros(len(bags_of_words))

    if "K = W_U" in labels[-1]: 
        labels = labels[:-1]
        continue

    # if "Q = W_E<" in labels[-1]: 
    #     labels = labels[:-1]

    # continue

    results = []

    queryside_vector = embeddings_dict[q_side_matrix][bags_of_words].T
    queryside_normalized = queryside_vector / (queryside_vector.var(dim=-1, keepdim=True) + eps).pow(0.5)
    query = W_Q.T @ queryside_normalized
    if USE_QUERY_BIAS:
        query += b_Q[:, None]

    keyside_vectors = embeddings_dict[k_side_matrix][torch.arange(d_vocab)]
    keyside_normalized = keyside_vectors / (keyside_vectors.var(dim=-1, keepdim=True) + eps).pow(0.5)
    key = keyside_normalized @ W_K
    if USE_KEY_BIAS:
        key += b_K[None]
    attention_scores = query.T @ key.T / np.sqrt(d_head)
    attention_scores = attention_scores.to("cpu")
    gc.collect()
    t.cuda.empty_cache()
    log_attentions_to_self = (attention_scores >= (attention_scores[torch.arange(len(bags_of_words)), bags_of_words])[:, None]).int().sum(dim=-1)

    all_log_attentions_to_self.append(log_attentions_to_self.cpu())
    sorted_log_attention = log_attentions_to_self.sort(descending=True).values
    lines.append(sorted_log_attention[sorted_log_attention.shape[0]//2]) # Median!
    print(lines[-1])

# In[21]:

# Make bar chart of distribution

relevant_labels = [
    'Q = W_U<br>K = W_EE',
    'Q = W_EE<br>K = W_EE',
    # 'Q = W_E<br>K = W_E', # TODO add this too...
]

relevant_distributions = [
    log_attentions_to_self_element[1].cpu() for log_attentions_to_self_element in enumerate(all_log_attentions_to_self) if labels[log_attentions_to_self_element[0]] in relevant_labels
]

#%%

hist(
    relevant_distributions,
    names = relevant_labels,
    labels={"variable": "Query and Key Inputs", "value": "Token rank"},
    width=600,
    height=600,
    nbins=50256,
    opacity=0.7,
    # marginal="box",
    template="simple_white",
    yaxis_type="log",
    return_fig=True,
).update_layout(
    xaxis=dict(range=[0, 20]),
    yaxis=dict(range=[-0.5, 5])
).show()

print("You ")

#%%

indices = [i for i, label in enumerate(labels) if "Q = W_E<" not in label]

the_lines = [lines[i] for i in indices]
the_labels = [labels[i] for i in indices]
the_log_attentions_to_self = [all_log_attentions_to_self[i] for i in indices]

#%%

# values_shown = lines
values_shown = [(x==1).int().sum().item() for x in the_log_attentions_to_self]

square_of_values = einops.rearrange(torch.tensor(values_shown), "(height width) -> height width", height=3)
# square_of_values = t.stack([square_of_values[:, 0], square_of_values[:, 2], square_of_values[:, 1]], dim=-1)
square_labels = square_of_values.tolist()
square_labels = [[int(square_label) for square_label in row] for row in square_labels]

fig = imshow(
    square_of_values.log(),
    # text_auto=True,
    title=f"Number of tokens that are top rank", #  with {USE_QUERY_BIAS=} {USE_KEY_BIAS=}",
    labels={"x": "Keyside lookup table", "y": "Queryside lookup table", "color": "Count"},
    x = ["W<sub>E</sub>", "W<sub>EE</sub>", "MLP<sub>0</sub>"], # sadly these are hardcoded
    y = ["W<sub>E</sub>", "W<sub>EE</sub>", "W<sub>U</sub>"],
    color_continuous_midpoint=None,
    range_color=(0, 10), # This manually defines the range of things
    coloraxis=dict(
        colorbar=dict(
        tickvals=np.log(np.array([1, 10, 100, 1000, 10000])),
        ticktext=['1', '10', '100', '1000', '10000']
    )),
    color_continuous_scale="Blues",
    return_fig=True,
)

for i, row in enumerate(square_labels):
    for j, label in enumerate(row):
        fig.add_annotation(
            x=j, # x-coordinate of the annotation
            y=i, # y-coordinate of the annotation
            text=str(round(label, 2)), # text label
            showarrow=False, # don't show an arrow pointing to the annotation
            # color="white" if "W_U" in the_labels[i] else "black",
            font=dict(
                color="white" if label > 1000 else "black",
            ),
        )

# Make it exactly square
fig.update_layout(
    width=400,
    height=400,
    margin=dict(
        l=0,
        r=0,
        b=0,
        # t=0,
        pad=0
    ),
)

fig.show()

# %%

# Save as JSON
# fig.write_json("quad_plot.json")
# fig.write_image("quad_plot2.pdf")

# %%
