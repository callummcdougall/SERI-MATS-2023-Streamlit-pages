# These are all the useful funcs I find myself using frequently, which aren't tied to a specific project.




# =============================================================================
# ! Imports & paths & constants
# =============================================================================

import sys, os
from pathlib import Path

for st_page_dir in [
    os.getcwd().split("SERI-MATS-2023-Streamlit-pages")[0] + "SERI-MATS-2023-Streamlit-pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("seri_mats_23_streamlit_pages")[0] + "seri_mats_23_streamlit_pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("seri-mats-2023-streamlit-pages")[0] + "seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
    os.getcwd().split("/app/seri-mats-2023-streamlit-pages")[0] + "/app/seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
    "/mount/src/seri-mats-2023-streamlit-pages/transformer_lens/rs/callum2/st_page",
]:
    if os.path.exists(st_page_dir):
        break
else:
    raise Exception("Couldn't find root dir")

root_dir = st_page_dir.replace("/transformer_lens/rs/callum2/st_page", "")

# We change to st_page_dir, so that we can read media (although maybe that's not necessary cause we have `ST_HTML_PATH` which we use directly)
os.chdir(st_page_dir)
ST_HTML_PATH = Path(st_page_dir) / "media"

# We make sure that the version of transformer_lens we can import from is 0th in the path
if sys.path[0] != root_dir: sys.path.insert(0, root_dir)



NEGATIVE_HEADS = [(10, 7), (11, 10)]

from transformer_lens.cautils.notebook import *


# =============================================================================
# ! String-parsing functions
# =============================================================================

def parse_str(s: str):
    doubles = "“”"
    singles = "‘’"
    for char in doubles: s = s.replace(char, '"')
    for char in singles: s = s.replace(char, "'")
    return s

def parse_str_tok_for_printing(s: str):
    s = s.replace("\n", "\\n")
    return s

def parse_str_toks_for_printing(s: List[str]):
    return list(map(parse_str_tok_for_printing, s))

def create_title_and_subtitles(
    title: str,
    subtitles: List[str],
) -> str:
    return f"{title}<br><span style='font-size:13px'>{'<br>'.join(subtitles)}</span>"

def process_webtext(
    seed: int,
    batch_size: int,
    seq_len: int,
    model: HookedTransformer,
    verbose: bool = False,
    return_indices: bool = False,
) -> Tuple[Int[Tensor, "batch seq"], List[List[str]]]:
    
    DATA_STR_ALL = get_webtext(seed=seed)
    DATA_STR_ALL = [parse_str(s) for s in DATA_STR_ALL]
    DATA_STR = []

    count = 0
    indices = []
    for i in range(len(DATA_STR_ALL)):
        num_toks = len(model.to_tokens(DATA_STR_ALL[i]).squeeze())
        if num_toks > seq_len:
            DATA_STR.append(DATA_STR_ALL[i])
            indices.append(i)
            count += 1
        if count == batch_size:
            break
    else:
        raise Exception("Couldn't find enough sequences of sufficient length.")

    DATA_TOKS = model.to_tokens(DATA_STR)
    DATA_STR_TOKS = model.to_str_tokens(DATA_STR)

    if seq_len < 1024:
        DATA_TOKS = DATA_TOKS[:, :seq_len]
        DATA_STR_TOKS = [str_toks[:seq_len] for str_toks in DATA_STR_TOKS]

    DATA_STR_TOKS_PARSED = list(map(parse_str_toks_for_printing, DATA_STR_TOKS))

    clear_output()
    if verbose:
        print(f"Shape = {DATA_TOKS.shape}\n")
        print("First prompt:\n" + "".join(DATA_STR_TOKS[0]))

    if return_indices: 
        return DATA_TOKS, DATA_STR_TOKS_PARSED, indices

    return DATA_TOKS, DATA_STR_TOKS_PARSED








# =============================================================================
# ! Statistical functions
# =============================================================================


def kl_div(
    logits1: Float[Tensor, "... d_vocab"],
    logits2: Float[Tensor, "... d_vocab"],
):
    '''
    Estimates KL divergence D_KL( logits1 || logits2 ), i.e. where logits1 is the "ground truth".

    Each tensor is assumed to have all dimensions be the batch dimension, except for the last one
    (which is a distribution over the vocabulary).

    In our use-cases, logits1 will be the non-ablated version of the model.
    '''

    logprobs1 = logits1.log_softmax(dim=-1)
    logprobs2 = logits2.log_softmax(dim=-1)
    logprob_diff = logprobs1 - logprobs2
    probs1 = logits1.softmax(dim=-1)

    return einops.reduce(
        probs1 * logprob_diff,
        "... d_vocab -> ...",
        reduction = "sum",
    )






# =============================================================================
# ! Utils functions for working with particular types of objects
# =============================================================================


def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()



def concat_dicts(d1: Dict[str, Tensor], d2: Dict[str, Tensor]) -> Dict[str, Tensor]:
    '''
    Given 2 dicts, return the dict of concatenated tensors along the zeroth dimension.

    Special case: if d1 is empty, we just return d2.

    Also, we make sure that d2 tensors are moved to cpu.
    '''
    if len(d1) == 0: return d2
    assert d1.keys() == d2.keys()
    return {k: t.cat([d1[k], d2[k]], dim=0) for k in d1.keys()}


def devices_are_equal(device_1: Union[str, t.device], device_2: Union[str, t.device]):
    '''
    Helper function, because devices "cuda:0" and "cuda" are actually the same.
    '''
    device_set = set([str(device_1), str(device_2)])
    
    return (len(device_set) == 1) or (device_set == {"cuda", "cuda:0"})


def first_occurrence(array_1D):
    series = pd.Series(array_1D)
    duplicates = series.duplicated(keep='first')
    inverted = ~duplicates
    return inverted.values

def first_occurrence_2d(tensor_2D):
    device = tensor_2D.device
    array_2D = utils.to_numpy(tensor_2D)
    return t.from_numpy(np.array([first_occurrence(row) for row in array_2D])).to(device)

def concat_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def make_list_correct_length(L, K, pad_tok: Optional[str] = None):
    '''
    If len(L) < K, pad list L with its last element until it is of length K.
    If len(L) > K, truncate.

    Special case when len(L) == 0, we just put the BOS token in it.
    '''
    if len(L) == 0:
        L = ["<|endoftext|>"]

    if pad_tok is None:
        pad_tok = L[-1]

    if len(L) <= K:
        L = L + [pad_tok] * (K - len(L))
    else:
        L = L[:K]

    assert len(L) == K
    return L

def update_mean(current_value: Tensor, new_value: Tensor, num_samples_so_far: int, num_new_samples: int):
    '''
    Updates a running mean.
    '''
    assert current_value.shape == new_value.shape, f"Shapes mismatch: old = {current_value.shape}, new = {new_value.shape}"
    return (num_samples_so_far * current_value + num_new_samples * new_value) / (num_samples_so_far + num_new_samples)


# =============================================================================
# ! Effective embeddings, and other model weight stuff
# =============================================================================

def get_effective_embedding(model: HookedTransformer, use_codys_without_attention_changes=True) -> Float[Tensor, "d_vocab d_model"]:
    # TODO - implement Neel's variation; attention to self from the token
    # TODO - make this consistent (i.e. change the func in `generate_bag_of_words_quad_plot` to also return W_U and W_E separately)

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

    resid_pre = W_E.unsqueeze(0)

    if not use_codys_without_attention_changes:
        pre_attention = model.blocks[0].ln1(resid_pre)
        attn_out = einops.einsum(
            pre_attention, 
            model.W_V[0],
            model.W_O[0],
            "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
        )
        resid_mid = attn_out + resid_pre
    else:
        resid_mid = resid_pre

    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    del resid_pre, pre_attention, attn_out, resid_mid, normalized_resid_mid, mlp_out
    t.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE,
        "W_U": W_U.T,
    }




# =============================================================================
# ! Projections (all versions, incl. old)
# =============================================================================

def gram_schmidt(vectors: Float[Tensor, "... d num"]) -> Float[Tensor, "... d num"]:
    '''
    Performs Gram-Schmidt orthonormalization on a batch of vectors, returning a basis.

    `vectors` is a batch of vectors. If it was 2D, then it would be `num` vectors each with length
    `d`, and I'd want a basis for these vectors in d-dimensional space. If it has more dimensions 
    at the start, then I want to do the same thing for all of them (i.e. get multiple independent
    bases).

    If the vectors aren't linearly independent, then some of the basis vectors will be zero (this is
    so we can still batch our projections, even if the subspace rank for each individual projection
    is not equal.
    '''
    # Make a copy of the vectors
    basis = vectors.clone()
    num_vectors = basis.shape[-1]
    
    # Iterate over each vector in the batch, starting from the zeroth
    for i in range(num_vectors):

        # Project the i-th vector onto the space orthogonal to the previous ones
        for j in range(i):
            projection = einops.einsum(basis[..., i], basis[..., j], "... d, ... d -> ...")
            basis[..., i] = basis[..., i] - einops.einsum(projection, basis[..., j], "..., ... d -> ... d")
        
        # Normalize this vector (we can set it to zero if it's not linearly independent)
        basis_vec_norm = basis[..., i].norm(dim=-1, keepdim=True)
        basis[..., i] = t.where(
            basis_vec_norm > 1e-4,
            basis[..., i] / basis_vec_norm,
            t.zeros_like(basis[..., i])
        )
    
    return basis

def project(
    vectors: Float[Tensor, "... d"],
    proj_directions: Float[Tensor, "... d num"],
    only_keep: Optional[Literal["pos", "neg"]] = None,
    gs: bool = True,
    return_coeffs: bool = False,
    return_perp: bool = False,
):
    '''
    `vectors` is a batch of vectors, with last dimension `d` and all earlier dimensions as batch dims.

    `proj_directions` is either the same shape as `vectors`, or has an extra dim at the end.

    If they have the same shape, we project each vector in `vectors` onto the corresponding direction
    in `proj_directions`. If `proj_directions` has an extra dim, then the last dimension is another 
    batch dim, i.e. we're projecting each vector onto a subspace rather than a single vector.
    '''
    # Sometimes proj_directions will be same shape as vectors, i.e. num=1
    if proj_directions.shape == vectors.shape:
        proj_directions = proj_directions.unsqueeze(-1)
    assert proj_directions.shape[:-1] == vectors.shape
    assert proj_directions.shape[-1] <= 30, "Shouldn't do too many vectors, GS orth might be computationally heavy I think?"

    # We might want to have done G-S orthonormalization first
    proj_directions_basis = gram_schmidt(proj_directions) if gs else proj_directions

    components_in_proj_dir = einops.einsum(
        vectors, proj_directions_basis,
        "... d, ... d num -> ... num"
    )
    if return_coeffs: return components_in_proj_dir
    
    if only_keep is not None:
        components_in_proj_dir = t.where(
            (components_in_proj_dir < 0) if (only_keep == "neg") else (components_in_proj_dir > 0),
            components_in_proj_dir,
            t.zeros_like(components_in_proj_dir)
        )

    vectors_projected = einops.einsum(
        components_in_proj_dir,
        proj_directions_basis,
        "... num, ... d num -> ... d"
    )

    return vectors_projected if not(return_perp) else (vectors_projected, vectors - vectors_projected)

# def project(
#     x: Float[Tensor, "... dim"],
#     dir: Union[List[Float[Tensor, "... dim"]], Float[Tensor, "... dim"]],
#     test: bool = False,
#     return_type: Literal["projections", "coeffs", "both"] = "projections",
# ):
#     '''
#     x: 
#         Shape (*batch_dims, d), or list of such shapes
#         Batch of vectors
    
#     dir:
#         Shape (*batch_dims, d)
#         Batch of vectors (which will be normalized)

#     test:
#         If true, runs a bunch of sanity-check-style tests, and prints out the output

#     Returns:
#         Two batches of vectors: x_dir and x_perp, such that:
#             x_dir + x_perp = x
#             x_dir is the component of x in the direction dir (or in the subspace
#             spanned by the vectors in dir, if dir is a list).

#     Notes:
#         Make sure x and dir (or each element in dir) have the same shape, I don't want to
#         mess up broadcasting by accident! Do einops.repeat on dir if you have to.
#     '''
#     assert return_type in ["projections", "coeffs", "both"]
#     device = x.device
#     if isinstance(dir, Tensor): dir = [dir]
#     assert all([x.shape == dir_.shape for dir_ in dir]), [x.shape, [d.shape for d in dir]]
#     dir = t.stack(dir, dim=-1)

#     # Get the SVD of the stack of matrices we're projecting in the direction of
#     # So U tells us directions, and V tells us linear combinations (which we don't need)
#     svd = t.svd(dir)
#     if test:
#         t.testing.assert_close(svd.U @ t.diag_embed(svd.S) @ svd.V.mH, dir)
#         U_norms = svd.U.norm(dim=-2) # norm of columns
#         t.testing.assert_close(U_norms, t.ones_like(U_norms))
#         # print("Running tests for projection function:")
#         # print("\tSVD tests passed")

#     # Calculate the component of x along the different directions of svd.U
#     x_coeffs = einops.einsum(
#         x, svd.U,
#         "... dim, ... dim directions -> ... directions"
#     )
#     if return_type == "coeffs":
#         return x_coeffs

#     # Project x onto these directions (summing over each of the directional projections)
#     x_dir = einops.einsum(
#         x_coeffs, svd.U,
#         "... directions, ... dim directions -> ... dim"
#     )

#     if test:
#         # First, test all the projections are orthogonal to each other
#         x_dir_projections = einops.einsum(
#             x_coeffs, svd.U,
#             "... directions, ... dim directions -> ... dim directions"
#         )
#         x_dir_projections_normed = x_dir_projections / x_dir_projections.norm(dim=-2, keepdim=True)
#         x_dir_cos_sims = einops.einsum(
#             x_dir_projections_normed, x_dir_projections_normed,
#             "... dim directions_left, ... dim directions_right -> ... directions_left directions_right"
#         )
        
#         x_dir_cos_sims_expected = t.eye(x_dir_cos_sims.shape[-1]).to(device)
#         diff = t.where(x_dir_cos_sims_expected.bool(), t.tensor(0.0).to(device), x_dir_cos_sims - x_dir_cos_sims_expected).abs().max().item()
#         assert diff < 1e-4, diff
#         # print(f"\tCos sim test passed: max cos sim diff = {diff:.4e}")

#         # Second, test that the sum of norms equals the original norm
#         x_dir_norms = x_dir.norm(dim=-1).pow(2)
#         x_dir_perp_norms = (x - x_dir).norm(dim=-1).pow(2)
#         x_norms = x.norm(dim=-1).pow(2)
#         diff = (x_dir_norms + x_dir_perp_norms - x_norms).abs().max().item()
#         assert diff < 1e-4, diff
#         # print(f"\tNorms test passed: max norm diff = {diff:.4e}")

#     if return_type == "both":
#         return x_dir, x - x_dir, x_coeffs
#     elif return_type == "projections":
#         return x_dir, x - x_dir

