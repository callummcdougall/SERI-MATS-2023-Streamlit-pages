# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import get_metric_from_end_state
import argparse

model: HookedTransformer = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)
model.set_use_hook_mlp_in(True)
model.set_use_attn_in(True)
DEVICE = "cuda"
SHOW_PLOT = True
DATASET_SIZE = 500
BATCH_SIZE = 20 # seems to be about the limit of what this box can handle
NUM_THINGS = 300
USE_RANDOM_SAMPLE = False
INDIRECT = True # disable for orig funcitonality

# Should we calculate KL to some model? Set to `None` if instead just considering loss reducing effects
GPT2_MODEL_FOR_KL: Optional[Literal["gpt2", "gpt2-xl"]] = "gpt2"

# %%

dataset = get_webtext(seed=17279)
max_seq_len = model.tokenizer.model_max_length

# %%

filtered_tokens = []
targets = []  # targets for prediction

print("Not rapid, but not THAT slow :-) ")
_idx = -1
while len(filtered_tokens) < DATASET_SIZE:
    _idx += 1
    cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
    if (
        len(cur_tokens) > max_seq_len
    ):  # so we're not biasing towards early sequence positions...
        filtered_tokens.append(cur_tokens[:max_seq_len])
        targets.append(cur_tokens[1 : max_seq_len + 1])

mybatch = torch.LongTensor(filtered_tokens[:BATCH_SIZE])
mytargets = torch.LongTensor(targets[:BATCH_SIZE])

#%%

if GPT2_MODEL_FOR_KL is not None: # TODO check this is working
    gpt2_for_kl = HookedTransformer.from_pretrained(GPT2_MODEL_FOR_KL)
    gpt2_for_kl_probs = t.zeros((BATCH_SIZE, max_seq_len, model.cfg.d_vocab))
    print("Starting GPT2-XL stuff")
    assert model.cfg.d_vocab == gpt2_for_kl.cfg.d_vocab, "Probably incompatible"
    for batch_idx in tqdm(range(BATCH_SIZE)):
        logits = gpt2_for_kl(mybatch[batch_idx : batch_idx + 1].to(DEVICE))[0]
        assert list(logits.shape) == [max_seq_len, gpt2_for_kl.cfg.d_vocab]
        gpt2_for_kl_probs[batch_idx] = t.nn.functional.log_softmax(logits, dim=-1).cpu()
        gc.collect()
        t.cuda.empty_cache()
    del gpt2_for_kl
    gc.collect()
    t.cuda.empty_cache()
    print("Done GPT2-XL stuff")

# %%

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
final_ln_scale_hook_name = "ln_final.hook_scale"
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name.endswith("hook_result")
    or name.endswith(".hook_resid_pre")
    or name == get_act_name("resid_mid", NEGATIVE_LAYER_IDX)
    or name == get_act_name("resid_pre", NEGATIVE_LAYER_IDX+1)
    or name == get_act_name("resid_mid", NEGATIVE_LAYER_IDX+1)
    or name == final_ln_scale_hook_name
    or "hook_scale" in name
)

model = model.to("cuda:0")
logits, cache = model.run_with_cache(
    mybatch.to("cuda:0"),
    names_filter=names_filter1,
    device="cpu",
)
model = model.to("cuda:0")
print("Done")
end_state = cache[END_STATE_HOOK].to("cuda")  # shape (batch_size, seq_len, hidden_size)
full_log_probs = torch.nn.functional.log_softmax(logits.cuda(), dim=-1).cpu()

del logits
gc.collect()
torch.cuda.empty_cache()

# %%

my_loss = get_metric_from_end_state(model, end_state.to(DEVICE), mytargets).cpu()

# %%

# see also the full test in arthurs_utils file
their_loss = model(
    mybatch.to(DEVICE),
    return_type="loss",
    loss_per_token=True,
).cpu()
assert list(their_loss.shape) == [
    my_loss.shape[0],
    my_loss.shape[1] - 1,
], f"their_loss.shape: {their_loss.shape}, my_loss.shape: {my_loss.shape}"
torch.testing.assert_close(
    their_loss,
    my_loss[:, :-1],
    atol=1e-2,
    rtol=1e-2,
)

# %%

results_log = {}

def setter_hook(z, hook, setting_value, setter_head_idx=None, intended_shape=None): # TODO unify assertions so adding signal to them is easier

    if setter_head_idx is not None:
        assert list(z.shape) == [BATCH_SIZE, max_seq_len, model.cfg.n_heads, model.cfg.d_model]
        z[:, :, setter_head_idx] = mean_output[None, None]

    else: 
        if intended_shape is None:
            assert list(z.shape) == [BATCH_SIZE, max_seq_len, model.cfg.d_model] == list(setting_value.shape), f"{z.shape=}, {setting_value.shape=} {[BATCH_SIZE, max_seq_len, model.cfg.d_model]}; {hook.name=}"
        else:
            assert list(z.shape) == intended_shape, (z.shape, intended_shape)

        z[:] = setting_value

    return z

def resetter_hook(z, hook, reset_value):
    assert list(z.shape) == [BATCH_SIZE, max_seq_len, model.cfg.d_model]
    z += reset_value
    return z

FREEZE_LN_ON_INTERMEDIATES = True
all_losses = {}

for FREEZE_LN in [False, True]:
    results_log = {}
    for layer_idx, head_idx in [(NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX)]:
        head_output_hook = f"blocks.{layer_idx}.attn.hook_result"
        head_output = cache[head_output_hook][
            :, :, head_idx
        ]# shape (batch_size, seq_len, hidden_size)
        mean_output = einops.reduce(
            head_output,
            "batch seq_len hidden_size -> hidden_size",
            reduction="mean",
        )
        mean_ablation_loss, mean_ablation_logits = get_metric_from_end_state(
            model=model,
            end_state=(end_state.cpu() - head_output + mean_output[None, None]).to(DEVICE),
            targets=mytargets,
            return_logits=True,
            frozen_ln_scale=cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
        )
        mean_ablation_loss = mean_ablation_loss.cpu()
        mean_ablation_log_probs = torch.nn.functional.log_softmax(mean_ablation_logits, dim=-1)
        del mean_ablation_logits
        gc.collect()
        t.cuda.empty_cache()
        
        if GPT2_MODEL_FOR_KL is not None:
            gc.collect()
            t.cuda.empty_cache()

            # also do a GPT2-XL experiment
            gpt2_kl = get_metric_from_end_state(
                model=model,
                end_state=end_state.cpu(),
                targets=None,
                return_logits=False,
                mode="kl",
                log_probs_reference=gpt2_for_kl_probs,
                device="cuda",
                frozen_ln_scale = cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
            )

            mean_ablation_kl = get_metric_from_end_state(
                model=model,
                end_state=(end_state.cpu() - head_output + mean_output[None, None]),
                targets=None,
                return_logits=False,
                mode="kl",
                log_probs_reference=gpt2_for_kl_probs,
                device="cuda",
                frozen_ln_scale = cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
            )

        if INDIRECT:
            # also do an indirect effect experiment
            
            model.reset_hooks()
            model.add_hook(
                head_output_hook,
                partial(setter_hook, setting_value=mean_output.to(DEVICE), setter_head_idx=head_idx),
            )
            _, indirect_cache = model.run_with_cache(
                mybatch.cuda(),
                names_filter=lambda name: name == END_STATE_HOOK,
            )
            model.reset_hooks()

            mean_ablated_total_loss = get_metric_from_end_state(
                model=model,
                end_state=indirect_cache[END_STATE_HOOK].to(DEVICE),
                targets=mytargets,
                return_logits=False,
                frozen_ln_scale = cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
            ).cpu()

            if FREEZE_LN:
                old_mean_ablated_indirect_loss = mean_ablated_indirect_loss.clone()

            mean_ablated_indirect_loss = get_metric_from_end_state(
                model=model,
                end_state=(indirect_cache[END_STATE_HOOK].cpu() + head_output - mean_output[None, None]).to(DEVICE),
                targets=mytargets,
                return_logits=False,
                frozen_ln_scale = cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
                compare_ln_scales = FREEZE_LN,
            )
            if FREEZE_LN:
                mean_ablated_indirect_loss, new_scales = mean_ablated_indirect_loss

            if FREEZE_LN:
                hist(
                    [cache[final_ln_scale_hook_name].cpu().flatten(), new_scales.cpu().flatten()],
                    names = ["Frozen", "Recomputed"],
                    width=1600,
                    height=600,
                    opacity=0.7,
                    marginal="box",
                    template="simple_white",
                )

            # Also freeze intermediate LNs
            # Empirically this does basically nothing lol though!
            model.reset_hooks()
            model.add_hook(
                head_output_hook,
                partial(setter_hook, setting_value=mean_output, setter_head_idx=head_idx),
            )
            for hook_name in [f"blocks.{layer_idx}.ln2.hook_scale" for layer_idx in range(NEGATIVE_LAYER_IDX, model.cfg.n_layers)] + [f"blocks.{layer_idx}.ln1.hook_scale" for layer_idx in range(NEGATIVE_LAYER_IDX+1, model.cfg.n_layers)]: # add freezing LN on the input hooks to the downstream MLPs and attention heads. We deal with the final LN in the get_metric function.
                model.add_hook(
                    hook_name,
                    partial(setter_hook, setting_value=cache[hook_name].to(DEVICE), setter_head_idx=None, intended_shape=[BATCH_SIZE, max_seq_len, model.cfg.n_heads, 1] if "ln1" in hook_name else [BATCH_SIZE, max_seq_len, 1]),
                )
            
            _, indirect_freeze_intermediate_cache = model.run_with_cache(
                mybatch.to("cuda:0"),
                names_filter=lambda name: name == END_STATE_HOOK,
            )
            model.reset_hooks()

            mean_ablated_indirect_freeze_intermediate_loss = get_metric_from_end_state(
                model=model,
                end_state=(indirect_freeze_intermediate_cache[END_STATE_HOOK].cpu() + head_output - mean_output[None, None]).to(DEVICE),
                targets=mytargets,
                return_logits=False,
                frozen_ln_scale = cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
            )

            if (NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX) == (layer_idx, head_idx) == (10, 7):
                model.reset_hooks()
                for cache_hook_name, dest_hook_name in [
                    (get_act_name("resid_mid", 10), get_act_name("mlp_in", 10)),
                    (get_act_name("resid_pre", 11), get_act_name("attn_in", 11)),
                    (get_act_name("resid_mid", 11), get_act_name("mlp_in", 11)),
                ]:
                    setting_value = cache[cache_hook_name] - head_output + mean_output

                    if "attn_in" in dest_hook_name:
                        setting_value = setting_value[:, :, None]

                    model.add_hook(
                        dest_hook_name,
                        partial(setter_hook, setting_value=setting_value, intended_shape=[BATCH_SIZE, max_seq_len, model.cfg.n_heads, model.cfg.d_model] if "attn_in" in dest_hook_name else None), # Need to deal with some broadcasting (alternatively could einops.repeat above)
                    )
                _, controlled_indirect_cache = model.run_with_cache(
                    mybatch.to("cuda:0"),
                    names_filter=lambda name: name == END_STATE_HOOK,
                )
                controlled_indirect_loss = get_metric_from_end_state(
                    model=model,
                    end_state=controlled_indirect_cache[END_STATE_HOOK],
                    targets=mytargets,
                    return_logits=False,
                    frozen_ln_scale = cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
                ).cpu()

                # Add the total version, except 11.10 sees normal stuff
                # (I hope that the loss is more than 50% of the way up to the direct effect loss)
                model.reset_hooks()
                model.add_hook(
                    head_output_hook,
                    partial(setter_hook, setting_value=mean_output, setter_head_idx=head_idx),
                )
                model.add_hook(
                    get_act_name("attn_in", 11), 
                    partial(setter_hook, setting_value=cache[get_act_name("resid_pre", 11)], setter_head_idx=10),
                )

                _, total_control_11 = model.run_with_cache(
                    mybatch.to("cuda:0"),
                    names_filter=lambda name: name == END_STATE_HOOK,
                )

                total_control_11_loss = get_metric_from_end_state(
                    model=model,
                    end_state=total_control_11[END_STATE_HOOK],
                    targets=mytargets,
                    return_logits=False,
                    frozen_ln_scale=cache[final_ln_scale_hook_name].to(DEVICE) if FREEZE_LN else None,
                ).cpu()

        loss_changes = (mean_ablation_loss - my_loss).cpu()
        flattened_loss_changes = einops.rearrange(
            loss_changes, "batch seq_len -> (batch seq_len)"
        )

        if SHOW_PLOT:
            assert INDIRECT

            all_losses = {
                **all_losses,
                "clean": my_loss,
                "mean_ablate_direct_effect" + ("_freeze" if FREEZE_LN else "") : mean_ablation_loss,
                "mean_ablate_all_effects"+ ("_freeze" if FREEZE_LN else ""): mean_ablated_total_loss,
                "mean_ablate_indirect_effects"+ ("_freeze" if FREEZE_LN else ""): mean_ablated_indirect_loss,
                "mean_ablated_indirect_freeze_intermediate_loss"+ ("_freeze" if FREEZE_LN else ""): mean_ablated_indirect_freeze_intermediate_loss,
            }

            if (NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX) == (layer_idx, head_idx) == (10, 7):
                all_losses["total_control_11_loss"+ ("_freeze" if FREEZE_LN else "")]=total_control_11_loss
                all_losses["controlled_indirect_loss"+ ("_freeze" if FREEZE_LN else "")]=controlled_indirect_loss

            if GPT2_MODEL_FOR_KL is not None:
                all_losses["gpt2_kl"+ ("_freeze" if FREEZE_LN else "")] = gpt2_kl
                all_losses["mean_ablation_kl"+ ("_freeze" if FREEZE_LN else "")] = mean_ablation_kl

all_losses_keys = list(all_losses.keys())
for key in all_losses_keys:
    all_losses[key] = einops.rearrange(
        all_losses[key], "batch seq_len -> (batch seq_len)"
    )
    print(key, all_losses[key].mean())

#%%

# nice figure that is about how there's correlation between the two interventions
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        # x = (mean_ablated_indirect_loss.cpu()-my_loss.cpu()).flatten(),
        # y = (old_mean_ablated_indirect_loss.cpu()-my_loss.cpu()).flatten(),
        x = cache[final_ln_scale_hook_name].cpu().numpy().flatten(),
        y = new_scales.cpu().numpy().flatten(),
        mode = "markers",
        # text = [f"{ii:.4f} {jj:.4f} {kk:.4f}" for ii, jj, kk in zip(mean_ablated_indirect_loss.cpu().flatten(), my_loss.cpu().flatten(), old_mean_ablated_indirect_loss.cpu().flatten(), strict=True)],
        text = [f"{ii:.4f} {jj:.4f}" for ii, jj in zip(cache[final_ln_scale_hook_name].cpu().numpy().flatten(), new_scales.cpu().numpy().flatten(), strict=True)],
        name = "Change in loss in individual example",
    )
)

fig.update_layout(
    xaxis_title = "Change in loss when we mean ablate indirect effects, and recompute LN",
    yaxis_title = "Change in loss when we mean ablate indirect effects, and freeze LN",
    height = 750,
)

# add y = x line
fig.add_trace(
    go.Scatter(
        x = [-2, 2],
        y = [-2, 2],
        mode = "lines",
        name = "y=x",
    )
)
fig.show()

#%%

if SHOW_PLOT:
    # filtered_all_losses = {key: all_losses[key] for key in all_losses_keys if any([key.startswith(thing) for thing in ["clean", "mean_ablate_indirect_effects", "mean_ablate_direct_effect"]])}
    filtered_all_losses = deepcopy(all_losses)

    sorted_losses = dict(sorted(filtered_all_losses.items(), key=lambda x: x[1].mean()))

    # actually I prefer a bar chart
    yvalues=[y.mean().item() for y in sorted_losses.values()]
    fig = px.bar(
        x=[str(x) for x in list(sorted_losses.keys())],
        y=yvalues,
        color = ["blue" for _ in range(len(sorted_losses)-1)] + ["red"],
        labels={
            "x": "10.7 Intervention",
            "y": "Average OWT Loss",
        },
        # error_y=[y.std().item()/np.sqrt(len(y)) for y in sorted_losses.values()], # TODO find a way to sample tons of points to drive down std
    )
    maxy = max(yvalues)
    miny = min(yvalues)
    fig.update_layout(
        yaxis_range=[miny - 0.1 * (maxy - miny), maxy + 0.1 * (maxy - miny)]
    )

    for inc in [0.001, 0.01]: # add line inc greater than clean loss, labellled inc increase
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=all_losses["clean"].mean().item() + inc,
            x1=len(sorted_losses) - 0.5,
            y1=all_losses["clean"].mean().item() + inc,
            line=dict(
                color="black",
                width=4,
                dash="dash",
            ),
        )
        fig.add_annotation(
            x=0,
            y=all_losses["clean"].mean().item() + inc,
            text=f"{inc} increase in loss",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
        )
    
    fig.show()
    normal_loss = all_losses["clean"]

results_log[(layer_idx, head_idx)] = {
    "mean_change_in_loss": flattened_loss_changes.mean().item(),
    "std": flattened_loss_changes.std().item(),
    "abs_mean": flattened_loss_changes.abs().mean().item(),
    "flattened_loss_changes": flattened_loss_changes.cpu(),
    "loss_changes": loss_changes.cpu(),
    "mean_ablation_loss": mean_ablation_loss.cpu(),
}

if GPT2_MODEL_FOR_KL is not None:
    results_log[(layer_idx, head_idx)]["gpt2_kl"] = gpt2_kl.cpu()
    results_log[(layer_idx, head_idx)]["mean_ablation_kl"] = mean_ablation_kl.cpu()
    results_log[(layer_idx, head_idx)]["gpt2_kl_change"] = (mean_ablation_kl - gpt2_kl).cpu()

print(list(results_log.items())[-1])

#%%

if GPT2_MODEL_FOR_KL is None:
    warnings.warn("There may be a crash here, but it is intended as below here the file is only GPT2-XL KL Divergence experiments")
    sys.exit(0) # rest of this file is for GPT2-XL...!

all_kls = (gpt2_kl).flatten()[:20000]
all_losses = (mean_ablation_loss - my_loss).flatten()[:20000]

indices = torch.argsort(all_losses)[:len(all_kls)//20]

px.scatter(
    x = all_kls[indices],
    y = all_losses[indices],
    labels = {
        "x": "KL Divergence to GPT-2",
        "y": "Change in GPT-2 Small loss when mean ablating 10.7",
    }
).show()


#%%

thing_used_as_mean_ablation = mean_ablation_loss if GPT2_MODEL_FOR_KL is None else mean_ablation_kl
thing_used_as_my_metric = my_loss if GPT2_MODEL_FOR_KL is None else gpt2_kl

# How much EV is explained by the direct effect of the head?
sorted_loss_change = torch.tensor(sorted(
    [
        (thing_used_as_mean_ablation[batch_idx, seq_idx].item() -
        thing_used_as_my_metric[batch_idx, seq_idx].item()) for batch_idx, seq_idx in itertools.product(
            range(BATCH_SIZE), range(max_seq_len)
        )
    ], 
    reverse=True,
))

useful_loss_changes = torch.nn.functional.relu(sorted_loss_change)
number_useful = useful_loss_changes.gt(0).sum().item()
FRACTION_DENOM = 20 # The denominator of the fraction of the tokens we will study
assert number_useful > len(sorted_loss_change) // FRACTION_DENOM, "Calcs won't make sense"

proportion_of_loss = useful_loss_changes[:len(sorted_loss_change)//FRACTION_DENOM].sum() / useful_loss_changes.sum()

if GPT2_MODEL_FOR_KL is not None:
    warnings.warn("We often say `loss` here when we're discussing KL divergence, essentially")

print(f"Average increase in loss from mean ablation of 10.7 direct effect *conditional on mean ablation being harmful* is\n{useful_loss_changes.sum().item() / number_useful=}\n")
print(f"Percentage of increase in loss contribution from the Top 1/{FRACTION_DENOM} is\n{proportion_of_loss*100 :.2f}%\n")

# I think this will explain 40% of the good loss
# Woo more than 50% : ) 

#%%

cumulative_useful_loss_changes = torch.cumsum(useful_loss_changes, dim=0)
fig = px.scatter(
    x=100* torch.tensor(range(len(cumulative_useful_loss_changes))) / len(cumulative_useful_loss_changes),
    y=100 * cumulative_useful_loss_changes / cumulative_useful_loss_changes[-1].item(),
)
fig.update_layout(
    title=f"Cumulative percentage of useful {'loss' if GPT2_MODEL_FOR_KL is None else 'KL divergence'} reduction explained by the direct effect of 10.7",
    xaxis_title="Percentage of tokens",
    yaxis_title="Percentage of loss explained",
)
fig.add_annotation(x=90, y=90,
    text=f"On these token completions, 10.7's direct effect increases {'loss' if GPT2_MODEL_FOR_KL is None else 'KL divergence'}",
    showarrow=True,
    arrowhead=1,
    ax=-10,
    ay=30,
)
fig.show()

#%%

# The global plot is weird 11.0 with crazy importance, 10.7 variance low ... ?
CAP = 10000
px.bar(
    x=[str(x) for x in list(results_log.keys())][:CAP],
    y=[x["mean_change_in_loss"] for x in results_log.values()][:CAP],
    error_y=[x["std"]/np.sqrt(len(results_log)) for x in results_log.values()][:CAP],
    title="Mean change in loss when mean ablating the direct effect of a head",
    labels = {"x": "Head", "y": "Mean change in loss"},
).show()

# %%

# Even accounting all the cases where heads are actively harmful, it still seems like we don't really get negative heads...
px.bar(
    x=[str(x) for x in list(results_log.keys())],
    y=[
        (x["loss_changes"] < 0).double().mean()
        for x in results_log.values()
    ],
    title="Proportion of token predictions in OWT where mean ablating the direct effect of a head is helpful",
).show()

#%%

props={}
for layer_idx, head_idx in results_log.keys():
    all_results = list(enumerate(results_log[(layer_idx, head_idx)]["flattened_loss_changes"]))
    sorted_results = sorted(
        all_results,
        key=lambda x: x[1].abs().item(),
        reverse=True,
    )
    cnt=0
    for _, loss_change in sorted_results[:len(sorted_results)//20]: # top 5 percent
        if loss_change<0: # good to mean ablate
            cnt+=1
    props[(layer_idx, head_idx)] = cnt/(len(sorted_results)//20)

#%%

px.bar(
    x=[str(x) for x in list(props.keys())],
    y=list(props.values()),
    title="Proportion of Top 5% absolute direct effect tokens where mean ablating the direct effect of a head is helpful",
).show()

#%%

def simulate_effective_embedding(
    model: HookedTransformer,
) -> Float[Tensor, "d_vocab d_model"]:
    """Cribbed from `transformer_lens/rs/callums_notebooks/subtract_embedding.ipynb`"""
    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    embeds = W_E.unsqueeze(0)
    pre_attention = model.blocks[0].ln1(embeds)
    # !!! b_O is not zero. Seems like b_V is, but we'll add it to be safe rather than sorry
    assert model.b_V[0].norm().item() < 1e-4
    assert model.b_O[0].norm().item() > 1e-4
    vout = (
        einops.einsum(  # equivalent to locking attention to 1
            pre_attention,
            model.W_V[0],
            "b s d_model, num_heads d_model d_head -> b s num_heads d_head",
        )
        + model.b_V[0]
    )
    post_attention = (
        einops.einsum(
            vout,
            model.W_O[0],
            "b s num_heads d_head, num_heads d_head d_model_out -> b s d_model_out",
        )
        + model.b_O[0]
    )
    resid_mid = post_attention + embeds
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()
    return {
        "W_U (or W_E, no MLPs)": W_U.T,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE,
    }
embeddings_dict = simulate_effective_embedding(model)

# %%

# Test that effective embedding is the same as lock attention and zero pos embed
model.reset_hooks()
model.add_hook(
    name="hook_pos_embed",
    hook=lambda z, hook: z * 0.0,
)
model.add_hook(
    name="blocks.0.attn.hook_pattern",
    hook=lock_attn,
)
mlp_out_hook = "blocks.0.hook_mlp_out"
hook_resid_pre = "blocks.1.hook_resid_pre"
_, cache_test = model.run_with_cache(
    torch.arange(model.tokenizer.model_max_length).unsqueeze(0).to(DEVICE),
    names_filter=lambda name: name in [mlp_out_hook, hook_resid_pre],
)
torch.testing.assert_close(
    cache_test[mlp_out_hook][0],
    embeddings_dict["W_E (only MLPs)"][: model.tokenizer.model_max_length],
    atol=1e-3,
    rtol=1e-3,
)
torch.testing.assert_close(
    cache_test[hook_resid_pre][0],
    embeddings_dict["W_E (including MLPs)"][: model.tokenizer.model_max_length],
    atol=1e-3,
    rtol=1e-3,
)

#%%

# (Deprecated some extra GPT-2 KL experiments...)