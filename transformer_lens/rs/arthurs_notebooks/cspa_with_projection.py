#%%

from transformer_lens.cautils.notebook import *

#%%



#%%

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
