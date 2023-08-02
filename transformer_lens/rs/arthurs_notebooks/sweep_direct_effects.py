#%%

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.arthurs_notebooks.arthur_utils import *
from transformer_lens.loading_from_pretrained import MODEL_ALIASES as MA
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default=None)
if ipython is not None:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

# %%

for MODEL_NAME in ["Pythia-410M"] + [
    model_name for model_name in MA.keys() if "gpt2" in model_name and ("small" in model_name or "medium" in model_name)
]:
    if args.model_name != None:
        MODEL_NAME = args.model_name
    print(MODEL_NAME)
    model = HookedTransformer.from_pretrained(MODEL_NAME)

    # %%

    mybatch, mytargets = get_filtered_webtext(
        model, 
        batch_size=30,
        max_seq_len=512, 
        dataset="stas/openwebtext-10k",
    )

    # %%

    logits = model(mybatch)

    #%%

    def get_loss(logits, mytargets):
        log_probs = logits.log_softmax(dim=-1)
        return -log_probs[
            torch.arange(mytargets.shape[0]).unsqueeze(1),
            torch.arange(mytargets.shape[1]).unsqueeze(0),
            mytargets,
        ].mean()

    #%%

    normal_loss = get_loss(logits, mytargets).item()

    #%%

    def mean_ablation_hook(z, hook, head_idx, value):
        z[:, :, head_idx, :] = value
        return z

    data = {}

    model.set_use_attn_result(True)
    for layer_idx in tqdm(range(model.cfg.n_layers)):
        for head_idx in range(model.cfg.n_heads):
            gc.collect()
            t.cuda.empty_cache()
            mean_head_output = model.run_with_cache(
                mybatch,
                names_filter = get_act_name("result", layer_idx),
            )[1][get_act_name("result", layer_idx)][:, :, head_idx, :]
            mean_head_output = mean_head_output.mean(dim=0)
            model.reset_hooks()
            model.add_hook(
                get_act_name("result", layer_idx),
                partial(mean_ablation_hook, head_idx=head_idx, value=mean_head_output),
            )
            logits = model(mybatch)
            loss = get_loss(logits, mytargets).item()

            data[(layer_idx, head_idx)] = (loss - normal_loss)/normal_loss

    #%%

    # with names equal to string of later and head idx
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[x[0] for x in data.keys()],
            y=[x for x in data.values()],
            text=[str(x) for x in data.keys()],    
            mode = "markers",        
        )
    )

    fig.update_layout(
        title="Average Logit Difference vs Layer and Head",
        xaxis_title="Layer",
        yaxis_title="Percentage loss increase",
    )

    # save fig

    fig.write_image(f"/root/TransformerLens/transformer_lens/rs/arthur/figures/{MODEL_NAME.replace('/','_')}_logit_diff.png")

    #%%

    MY_FNAME = "/root/TransformerLens/transformer_lens/rs/arthur/json_data/head_survey.json"
    with open(MY_FNAME, "r") as f:
        cur_json = json.load(f)

    # update the data
    cur_json[model.cfg.model_name] = {str(k): v for k, v in data.items()}

    # write the data
    # write the updated data (don't overwrite!)
    with open(MY_FNAME, "w") as f:
        f.write(json.dumps(cur_json, indent=4))

    if args.model_name != None:
        break


# %%

if ipython is not None:
    fig = go.Figure()
    json_file = "/root/TransformerLens/transformer_lens/rs/arthur/json_data/head_survey.json"
    with open(json_file, "r") as f:
        cur_json = json.load(f)

    for model_name, data in cur_json.items():
        if "alias" in model_name:# or "battlestar" in model_name:
            continue
        fig.add_trace(
            go.Scatter(
                x=[int(x.split()[0][1:-1]) for x in data.keys()],
                y=[x for x in data.values()],
                text=[str(x) for x in data.keys()],    
                mode = "markers",        
                name=model_name if "gpt2-small" not in model_name else "stanford_gpt2_small",
                marker=dict(
                    size=12 if "gpt2-small" in model_name or "gpt2" == model_name or "gpt2-medium" == model_name else 6,
                ),
            )
        )

    fig.update_layout(
        title="Total effect of mean ablating attention heads. Fatter dots are models trained with dropout",
        xaxis_title="Layer of head",
        yaxis_title="Percentage loss increase",
    )

    fig.show()
# %%
