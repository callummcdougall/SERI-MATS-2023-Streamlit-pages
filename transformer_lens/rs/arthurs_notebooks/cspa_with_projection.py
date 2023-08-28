# %%
"""
CSPA with projection
"""

# %%

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum2.cspa.cspa_functions import (
    FUNCTION_STR_TOKS,
    get_cspa_results,
    get_cspa_results_batched,
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
    add_cspa_to_streamlit_page,
    show_graphs_and_summary_stats,
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

# %%

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

# %%

BATCH_SIZE = 500 # 80 for viz
SEQ_LEN = 1000 # 61 for viz

current_batch_size = 17 # These are smaller values we use for vizualization since only these appear on streamlit
current_seq_len = 61

NEGATIVE_HEADS = [(10, 7), (11, 10)]
DATA_TOKS, DATA_STR_TOKS_PARSED, indices = process_webtext(seed=6, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, model=model, verbose=True, return_indices=True)
cspa_semantic_dict = {}

# Calculate mean on a different dataset
HELD_OUT_DATA_TOKS, HELD_OUT_DATA_STR_TOKS_PARSED = process_webtext(seed=419287, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, model=model, verbose=False, return_indices=False)

# %%

"""
## Running CSPA code
"""

# %%
"""
### `K_u = 0.05`, no semantic similarity, batch size = 100
"""

# %%

result_mean = get_result_mean([(10, 7), (11, 10)], HELD_OUT_DATA_TOKS, model, 
verbose=True)

# %%

# Let's see how the variance changes with different batch sizes

cspa_results_qk_ov = get_cspa_results_batched(
    model = model,
    toks = DATA_TOKS,
    max_batch_size = 50, # 50,
    negative_head = (10, 7),
    interventions = ["qk", "ov"],
    K_unembeddings = 0.05, # most interesting in range 3-8 (out of 80)
    K_semantic = 1, # either 1 or up to 8 to capture all sem similar
    semantic_dict = cspa_semantic_dict,
    result_mean = result_mean,
    use_cuda = False,
    verbose = True,
    compute_s_sstar_dict = False,
)

# %%

show_graphs_and_summary_stats(cspa_results_qk_ov)

# %%
for fig_name, fig in zip(
    ["loss", "loss-absolute", "kl-div"],
    [fig_line_loss, fig_line_loss_absolute, fig_line_kl], 
    strict=True,
):
    fig.write_image(fig_name + ".pdf")

# %%
"""
# Is there a correlation between the absolute value of 10.7 effect on loss and 10.7's effect on increasing KL Divergence from the model?
"""

# %%
print(BATCH_SIZE, SEQ_LEN, current_batch_size, current_seq_len, len(indices))

# %%
VIZ_BATCH_SIZE = QK_OV_BATCH_SIZE # change to current_batch_size for a smaller subset that can be streamlit vizualized
VIZ_SEQ_LEN = SEQ_LEN -1 # change to current_seq_len for a smaller subset that can be streamlit vizualized
BATCH_INDICES = torch.arange(VIZ_BATCH_SIZE) # (change to indices to filter for correct vals on streamlit)

batch_indices = (BATCH_INDICES[:VIZ_BATCH_SIZE].unsqueeze(-1) + torch.zeros(VIZ_SEQ_LEN).unsqueeze(0))
seq_indices = (torch.zeros(VIZ_BATCH_SIZE).unsqueeze(-1) + torch.arange(VIZ_SEQ_LEN).unsqueeze(0))

my_dict = {
    "Mean Ablate 10.7 Loss - Normal Loss": cspa_results_qk_ov["loss_ablated"].clone().cpu() - cspa_results_qk_ov["loss"].clone().cpu(),
    "Mean Ablation KL Divergence to Model": cspa_results_qk_ov["kl_div_ablated_to_orig"][:, :-1].clone().cpu(),
    "CSPA Loss - Normal Loss": cspa_results_qk_ov["loss_cspa"].clone().cpu() - cspa_results_qk_ov["loss"].clone().cpu(),
    "CSPA KL": cspa_results_qk_ov["kl_div_cspa_to_orig"][:, :-1].clone().cpu(),
}

for k in list(my_dict.keys()):
    print(my_dict[k].shape)
    my_dict[k] = my_dict[k][:VIZ_BATCH_SIZE, :VIZ_SEQ_LEN].flatten()
    print(my_dict[k].shape)

my_dict["Batch Indices"] = batch_indices.flatten().float()
my_dict["Seq Indices"] = seq_indices.flatten().float()

print(my_dict["Batch Indices"].shape)
print(my_dict["Seq Indices"].shape)

df = pd.DataFrame(my_dict)

# %%
# assert all([v.shape == (current_batch_size*current_seq_len,) for _, v in my_dict.items()])

# %%
# import warnings; warnings.warn("Check out color when working")
# fig = go.Figure()
# fig.add_trace(

color = "CSPA KL"
px.scatter(
    df,
    x = "Mean Ablate 10.7 Loss - Normal Loss",
    y = "Mean Ablation KL Divergence to Model",
    hover_data = ["Batch Indices", "Seq Indices"],
    color = color,
    color_continuous_scale = "Blues" if "KL" in color else "RdBu_r",
    color_continuous_midpoint=0.0,
    range_color=((0.0, 0.09) if "KL" in color else None),
).update_traces(
    marker=dict(
        line=dict(
            color='black', # Border color
            width=0.5  # Border width
        )
    )
).show()

# %%
import plotly.graph_objects as go
import pandas as pd

Q=30
SQUARE = False

# Create quantile labels and bins
df['quantile'], bins = pd.qcut(df['CSPA Loss - Normal Loss'].abs()**(2 if SQUARE else 1), Q, labels=False, retbins=True)

# Initialize figure
fig = go.Figure()

def interpolate_colors(N):
    red = np.array([1, 0, 0])
    blue = np.array([0, 0, 1])
    colors = []
    
    for i in range(N):
        ratio = i / (N - 1)
        interpolated_color = (1 - ratio) * red + ratio * blue
        hex_color = "#{:02x}{:02x}{:02x}".format(int(interpolated_color[0]*255), int(interpolated_color[1]*255), int(interpolated_color[2]*255))
        colors.append(hex_color)
        
    return colors


colors = interpolate_colors(Q)
xlabel = 'CSPA Loss - Normal Loss'
ylabel = 'CSPA KL'

# Add scatter traces for each quantile
for i, q in enumerate(sorted(df['quantile'].unique())):
    subset = df[df['quantile'] == q]
    min_val = round(bins[q], 2)
    max_val = round(bins[q + 1], 2)
    fig.add_trace(
        go.Scatter(
            x=subset[xlabel].abs()**(2 if SQUARE else 1),
            y=subset[ylabel],
            mode='markers',
            name=f'CSPA quantile {min_val} to {max_val}, average contribution {round(float(np.array(subset["CSPA Loss - Normal Loss"]).mean()), 4)}',
            marker=dict(color=colors[i])
        )
    )

# Update layout
fig.update_layout(
    height=1000,
    xaxis_title=xlabel,
    yaxis_title=ylabel,
)

xs = df[xlabel].abs()**(2 if SQUARE else 1)
ys = df[ylabel]

r2 = np.corrcoef(xs, ys)[0, 1] ** 2
# get best fit line 
m, b = np.polyfit(xs, ys, 1)

# add best fit line from min x to max x
fig.add_trace(
    go.Scatter(
        x=[min(xs), max(xs)],
        y=[m * min(xs) + b, m * max(xs) + b],
        mode="lines",
        name=f"r^2 = {r2:.3f}",
    )
)

# add y = mx + c label
fig.add_annotation(
    x=0.1,
    y=0.1,
    text=f"y = {m:.3f}x + {b:.3f}",
    showarrow=False,
)

for i in range(Q):
    fig.data[i].visible = "legendonly"
fig.update_layout(title="Title CSPA correlation on KL and Loss. Click on the various quartiles to see the distributions in these quartiles" + ("Using squaring" if SQUARE else ""))

fig.show()

print("Average increase in loss from CSPA", cspa_results_qk_ov["loss_cspa"].mean() - cspa_results_qk_ov["loss"].mean())
print(f"Which should be the same as the average of the above: {df['CSPA Loss - Normal Loss'].mean()}")

# %%
"""
Answer: the correlation is pretty weak. The good news is that there do not appear to be many points that loss systematically misses but KL captures.

The line of best fit being 

Change in KL = 0.067 * Absolute change in loss

with no constant factor is nice, though R^2 = 0.34 is low (this isn't surprising, the quantities are measuring different things).
"""

# %%
Q = 50

x_std = df['CSPA Loss - Normal Loss'].std()
y_std = df['CSPA KL'].std()

x_min = - x_std # df['CSPA Loss - Normal Loss'].min()
x_max = x_std # df['CSPA Loss - Normal Loss'].max()
y_min = 0.0
y_max = y_std # df['CSPA KL'].max()

heatmap_vals = torch.zeros(Q, Q)

for x_quantile in range(Q):
    for y_quantile in range(Q):
        x_subset = df['CSPA Loss - Normal Loss'] >= x_min + (x_max - x_min) * x_quantile / Q 
        x_subset = x_subset & (df['CSPA Loss - Normal Loss'] <= x_min + (x_max - x_min) * (x_quantile+1) / Q)
        
        y_subset = df['CSPA KL'] >= y_min + (y_max - y_min) * y_quantile / Q 
        y_subset = y_subset & (df['CSPA KL'] <= y_min + (y_max - y_min) * (y_quantile+1) / Q)

        heatmap_size = (x_subset & y_subset).to_numpy().astype("int").sum() / len(df['CSPA KL'])
        heatmap_vals[x_quantile, y_quantile] = np.log(heatmap_size) # Can use log here...

fig = imshow(
    heatmap_vals[:, torch.arange(heatmap_vals.shape[0]-1, -1, -1)].T, # Does two things: makes axes the right way around, and in my opinion heatmaps x and y are the wrong way round
    title = f"Log Density of Points in CSPA Ranges",
    width = 500, 
    height = 500,
    labels = {"x": "CSPA Loss - Model Loss", "y": "CSPA KL"},
    x = [str(round(x_min + (x_max - x_min) * x_quantile / Q, 4)) for x_quantile in range(Q)],
    y = [str(round(y_min + (y_max - y_min) * y_quantile / Q, 5)) for y_quantile in range(Q)][::-1],
    # text_auto = ".2f",
    range_color=(heatmap_vals.min().item(), heatmap_vals.max().item()),
    color_continuous_scale="Blues",
    return_fig=True,
    color_continuous_midpoint=None,
)

# Set background color to white
fig.update_layout(
    paper_bgcolor='rgba(255,255,255,255)',
    plot_bgcolor='rgba(255,255,255,255)'
)
fig.show()

print(f"{df['CSPA Loss - Normal Loss'].std()=} {df['CSPA KL'].std()=}")


# %%
fig.write_image("Densities.pdf")

# %%
"""
### `K_u = 0.05`, no semantic similarity, batch size = 500
"""

# %%
cspa_results_qk_ov = get_cspa_results_batched(
    model = model,
    toks = DATA_TOKS[:, :], # [:50],
    max_batch_size = 5, # 50,
    negative_head = (10, 7),
    interventions = ["qk", "ov"],
    K_unembeddings = 0.05, # most interesting in range 3-8 (out of 80)
    K_semantic = 1, # either 1 or up to 8 to capture all sem similar
    only_keep_negative_components = True,
    semantic_dict = cspa_semantic_dict,
    result_mean = result_mean,
    use_cuda = False,
    verbose = True,
    compute_s_sstar_dict = False,
)

# fig_dict = generate_scatter(cspa_results_qk_ov, DATA_STR_TOKS_PARSED, batch_index_colors_to_highlight=[51, 300])
fig_loss_line = generate_loss_based_scatter(cspa_results_qk_ov, nbins=200, values="loss")
fig_loss_line_kl = generate_loss_based_scatter(cspa_results_qk_ov, nbins=200, values="kl-div")

kl_div_ablated_to_orig = cspa_results_qk_ov["kl_div_ablated_to_orig"].mean()
kl_div_cspa_to_orig = cspa_results_qk_ov["kl_div_cspa_to_orig"].mean()

print(f"Mean KL divergence from ablated to original: {kl_div_ablated_to_orig:.4f}")
print(f"Mean KL divergence from CSPA to original: {kl_div_cspa_to_orig:.4f}")
print(f"Ratio = {kl_div_cspa_to_orig / kl_div_ablated_to_orig:.3f}")
print(f"Performance explained = {1 - kl_div_cspa_to_orig / kl_div_ablated_to_orig:.3f}")
ma_max = fig_loss_line_kl.data[0].x[-1]
cspa_max = fig_loss_line_kl.data[0].y[-1]
print(f"Most extreme quantile: fraction explained = 1 - ({cspa_max:.3f}/{ma_max:.3f}) = {1 - cspa_max/ma_max:.3f}")

# %%
"""
# How well does this metric do for other heads?
"""

# %%
result_mean = get_result_mean([
    (layer, head)
    for layer in [8, 9, 10, 11] for head in range(12)
], DATA_TOKS[:100, :], model, verbose=True)

# result_mean = get_result_mean([(10, 7), (11, 10)], DATA_TOKS[:100, :], model, verbose=True)

# %%
kl_results = t.zeros(2, 4, 12).to(device)
loss_results = kl_results.clone()
normed_loss_results = kl_results.clone()
non_normed_loss_results = kl_results.clone()
squared_loss_results = kl_results.clone()

for i, only_keep_negative_components in enumerate([True, False]):

    for layer, head in tqdm(list(itertools.product([8, 9, 10, 11], range(12)))):

        cspa_results_qk_ov = get_cspa_results_batched(
            model = model,
            toks = DATA_TOKS[:50, :200], # [:50],
            max_batch_size = 1, # 50,
            negative_head = (layer, head),
            interventions = ["qk", "ov"],
            K_unembeddings = 0.05, # most interesting in range 3-8 (out of 80)
            K_semantic = 1, # either 1 or up to 8 to capture all sem similar
            only_keep_negative_components = only_keep_negative_components,
            semantic_dict = cspa_semantic_dict,
            result_mean = result_mean,
            use_cuda = True,
            verbose = False,
            compute_s_sstar_dict = False,
        )

        kl_div_ablated_to_orig = cspa_results_qk_ov["kl_div_ablated_to_orig"].mean().item()
        kl_div_cspa_to_orig = cspa_results_qk_ov["kl_div_cspa_to_orig"].mean().item()

        diff_of_loss_ablated_to_orig = (cspa_results_qk_ov["loss_ablated"] - cspa_results_qk_ov["loss"])
        squared_loss_diff = (diff_of_loss_ablated_to_orig**2).mean().item()
        normed_loss_diff = diff_of_loss_ablated_to_orig.abs().mean().item()
        non_normed_loss_diff = diff_of_loss_ablated_to_orig.mean().item()

        diff_of_loss_cspa_to_orig = (cspa_results_qk_ov["loss_cspa"] - cspa_results_qk_ov["loss"])
        normed_cspa_loss_diff = diff_of_loss_cspa_to_orig.abs().mean().item() 
        squared_cspa_loss_diff = (diff_of_loss_cspa_to_orig**2).mean().item()
        non_normed_cspa_loss_diff = diff_of_loss_cspa_to_orig.mean().item()

        kl_performance_explained = 1 - kl_div_cspa_to_orig / kl_div_ablated_to_orig
        kl_results[i, layer - 8, head] = kl_performance_explained

        normed_loss_performance_explained = 1 - normed_cspa_loss_diff / normed_loss_diff
        normed_loss_results[i, layer - 8, head] = normed_loss_performance_explained

        squared_loss_performance_explained = 1 - squared_cspa_loss_diff / squared_loss_diff
        squared_loss_results[i, layer - 8, head] = squared_loss_performance_explained

        non_normed_loss_performance_explained = 1 - non_normed_cspa_loss_diff / non_normed_loss_diff
        non_normed_loss_results[i, layer - 8, head] = non_normed_loss_performance_explained

# %%
for results_name, results in zip(["KL", "Net effect on loss", "Absolute difference in loss", "Squared effect on loss"], [kl_results, non_normed_loss_results, normed_loss_results, squared_loss_results], strict=True):
    imshow(
        results,
        facet_col = 0,
        facet_labels = ["Only keep negative components", "Keep negative and positive components"],
        title = f"{results_name} performance of head explained by CSPA",
        width = 1800, 
        height = 450,
        labels = {"x": "Head", "y": "Layer"},
        y = [str(i) for i in [8, 9, 10, 11]],
        text_auto = ".2f",
        range_color=[0,1],
        color_continuous_scale="Blues",
    )

# %%
"""
### Umm ... what?
"""

# %%
cspa_results_qk_ov = get_cspa_results_batched(
    model = model,
    toks = DATA_TOKS[:80, :200], # [:50],
    max_batch_size = 2, # 50,
    negative_head = (10, 1),
    interventions = ["qk", "ov"],
    K_unembeddings = 0.05, # most interesting in range 3-8 (out of 80)
    K_semantic = 1, # either 1 or up to 8 to capture all sem similar
    only_keep_negative_components = False,
    semantic_dict = cspa_semantic_dict,
    result_mean = result_mean,
    use_cuda = True,
    verbose = True,
    compute_s_sstar_dict = False,
    keep_self_attn = False,
)

# fig_dict = generate_scatter(cspa_results_qk_ov, DATA_STR_TOKS_PARSED, batch_index_colors_to_highlight=[51, 300])
fig_loss_line = generate_loss_based_scatter(cspa_results_qk_ov, nbins=200, values="loss")
fig_loss_line_kl = generate_loss_based_scatter(cspa_results_qk_ov, nbins=200, values="kl-div")

kl_div_ablated_to_orig = cspa_results_qk_ov["kl_div_ablated_to_orig"].mean()
kl_div_cspa_to_orig = cspa_results_qk_ov["kl_div_cspa_to_orig"].mean()

print(f"Mean KL divergence from ablated to original: {kl_div_ablated_to_orig:.4f}")
print(f"Mean KL divergence from CSPA to original: {kl_div_cspa_to_orig:.4f}")
print(f"Ratio = {kl_div_cspa_to_orig / kl_div_ablated_to_orig:.3f}")
print(f"Performance explained = {1 - kl_div_cspa_to_orig / kl_div_ablated_to_orig:.3f}")
ma_max = fig_loss_line_kl.data[0].x[-1]
cspa_max = fig_loss_line_kl.data[0].y[-1]
print(f"Most extreme quantile: fraction explained = 1 - ({cspa_max:.3f}/{ma_max:.3f}) = {1 - cspa_max/ma_max:.3f}")

# %%
"""
### The actual code which appears on the dedicated streamlit page:
"""

# %%
cspa_results_qk_ov, s_sstar_pairs_qk_ov = get_cspa_results_batched(
    model = model,
    toks = DATA_TOKS, # [:50],
    max_batch_size = 60, # 50,
    negative_head = (10, 7),
    interventions = ["qk", "ov"],
    K_unembeddings = 5,
    K_semantic = 1,
    only_keep_negative_components = True,
    semantic_dict = cspa_semantic_dict,
    use_cuda = True,
    verbose = True,
    compute_s_sstar_dict = True,
)
# TODO - figure out where the bottleneck is via line profiler. I thought it was projections, but now it seems like this is not the case
# Seems like it's this func: get_top_predicted_semantically_similar_tokens
# %load_ext line_profiler
# %lprun -f func func(arg, kwarg=kwarg)

fig_dict = generate_scatter(cspa_results_qk_ov, DATA_STR_TOKS_PARSED, batch_index_colors_to_highlight=[51, 300])
fig_loss_line = generate_loss_based_scatter(cspa_results_qk_ov, nbins=200, values="loss")
fig_loss_line_kl = generate_loss_based_scatter(cspa_results_qk_ov, nbins=200, values="kl-div")

kl_div_ablated_to_orig = cspa_results_qk_ov["kl_div_ablated_to_orig"].mean()
kl_div_cspa_to_orig = cspa_results_qk_ov["kl_div_cspa_to_orig"].mean()
print(f"Mean KL divergence from ablated to original: {kl_div_ablated_to_orig:.4f}")
print(f"Mean KL divergence from CSPA to original: {kl_div_cspa_to_orig:.4f}")
print(f"Ratio = {kl_div_cspa_to_orig / kl_div_ablated_to_orig:.3f}")
print(f"Performance explained = {1 - kl_div_cspa_to_orig / kl_div_ablated_to_orig:.3f}")

# %%
"""
# Adding CSPA to the Streamlit page ("Browse Examples")

This code adds the CSPA plots to the HTML plots for the Streamlit page. It creates a 5th tab called `CSPA`, and adds to the logit and DLA plots in the second tab (the latter is mainly for our use, while we're iterating on and improving the CSPA code).

I've added to this code in a pretty janky way, so that it can show more than one CSPA plot stacked on top of each other.
"""

# %%
cspa_results, s_sstar_pairs = get_cspa_results_batched(
    model = model,
    toks = DATA_TOKS[:48, :61], # [:50],
    max_batch_size = 2, # 50,
    negative_head = (10, 1),
    interventions = ["qk", "ov"],
    K_unembeddings = 0.05, # most interesting in range 3-8 (out of 80)
    K_semantic = 1, # either 1 or up to 8 to capture all sem similar
    only_keep_negative_components = False,
    semantic_dict = cspa_semantic_dict,
    result_mean = result_mean,
    use_cuda = True,
    verbose = True,
    compute_s_sstar_dict = True,
    return_dla = True,
    return_logits = True,
    keep_self_attn = True,
)

# %%
add_cspa_to_streamlit_page(
    cspa_results = cspa_results,
    s_sstar_pairs = s_sstar_pairs,
    html_plots_filename = f"GZIP_HTML_PLOTS_b48_s61.pkl",
    data_str_toks_parsed = [s[:61] for s in DATA_STR_TOKS_PARSED[:48]],
    toks_for_doing_DLA = DATA_TOKS[:48, :61],
    model = model,
    verbose = True,
    # test_idx = 36,
)

# %%
# b = 51
# add_cspa_to_streamlit_page(
#     cspa_results = cspa_results,
#     s_sstar_pairs = s_sstar_pairs,
#     html_plots_filename = f"GZIP_HTML_PLOTS_b{b}_s61.pkl",
#     data_str_toks_parsed = DATA_STR_TOKS_PARSED,
#     toks_for_doing_DLA = DATA_TOKS,
#     model = model,
#     verbose = True,
#     # test_idx = 32,
# )

# b = 200
# add_cspa_to_streamlit_page(
#     cspa_results = {"k=4": cspa_results, "k=1": cspa_results_1},
#     s_sstar_pairs = {"k=4": s_sstar_pairs, "k=1": s_sstar_pairs_1},
#     html_plots_filename = f"GZIP_HTML_PLOTS_b{b}_s61.pkl",
#     data_str_toks_parsed = DATA_STR_TOKS_PARSED,
#     toks_for_doing_DLA = DATA_TOKS,
#     model = model,
#     verbose = True,
#     # test_idx = 32,
# )

# %%
def cos_sim_of_toks(
    toks1: List[str],
    toks2: List[str],
):
    U1 = model.W_U.T[model.to_tokens(toks1, prepend_bos=False).squeeze()]
    U2 = model.W_U.T[model.to_tokens(toks2, prepend_bos=False).squeeze()]

    if U1.ndim == 1: U1 = U1.unsqueeze(0)
    if U2.ndim == 1: U2 = U2.unsqueeze(0)

    U1_normed = U1 / t.norm(U1, dim=-1, keepdim=True)
    U2_normed = U2 / t.norm(U2, dim=-1, keepdim=True)

    imshow(
        U1_normed @ U2_normed.T,
        title = "Cosine similarity of unembeddings",
        x = toks2,
        y = toks1,
    )

cos_sim_of_toks(
    [" stuff"],
    [" devices", " phones", " screens", " device", " phone", " Android"]
)

# %%
"""
# Testing the code for "love and war"
"""

# %%
result_mean_as_tensor = t.load(ST_HTML_PATH / "result_mean.pt")
result_mean = {(10, 7): result_mean_as_tensor[0], (11, 10): result_mean_as_tensor[1]}

prompt = "I picked up the first box. I picked up the second box. I picked up the third and final box."
toks = model.to_tokens(prompt)
str_toks = model.to_str_tokens(toks)
if isinstance(str_toks[0], str): str_toks = [str_toks]
# Parse the string tokens for printing
str_toks_parsed = [list(map(parse_str_tok_for_printing, s)) for s in str_toks]

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
)
cspa_results, s_sstar_pairs = get_cspa_results(
    model=model,
    toks=toks,
    negative_head=(10, 7), #  this currently doesn't do anything; it's always 10.7
    components_to_project=["o"],
    K_unembeddings=5,
    K_semantic=3,
    semantic_dict=cspa_semantic_dict,
    effective_embedding="W_E (including MLPs)",
    result_mean=result_mean,
    use_cuda=False,
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
    test_idx=0,
)

# %%
"""
# Modular CSPA
"""

# %%
OV_BATCH_SIZE = 50

cspa_results_qk = get_cspa_results_batched(
    model = model,
    toks = DATA_TOKS[:OV_BATCH_SIZE],
    max_batch_size = 1, # 50,
    negative_head = (10, 7),
    interventions = ["qk"],
    K_unembeddings = 0.05, # most interesting in range 3-8 (out of 80)
    K_semantic = 1, # either 1 or up to 8 to capture all sem similar
    semantic_dict = cspa_semantic_dict,
    result_mean = result_mean,
    use_cuda = False,
    verbose = True,
    compute_s_sstar_dict = False,
)
clear_output() # Weird cell, it hogs space

# %%
show_graphs_and_summary_stats(cspa_results_qk)