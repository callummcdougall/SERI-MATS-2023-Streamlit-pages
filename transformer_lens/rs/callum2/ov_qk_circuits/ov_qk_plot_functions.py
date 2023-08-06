from transformer_lens.cautils.utils import *

def plot_logit_lens(
    points_to_plot,
    resid_pre_head: ActivationCache,
    model: HookedTransformer,
    DATA_STR_TOKS_PARSED: List[List[str]],
    neg: bool = False,
    k: int = 15,
    title: Optional[str] = None,
):

    logits: Float[Tensor, "batch seq d_vocab"] = resid_pre_head @ model.W_U

    for seq, pos, expected in points_to_plot:
        if isinstance(expected, str):
            expected = [expected]
        s = f"{''.join(DATA_STR_TOKS_PARSED[seq][pos-4:pos+1])!r} -> {DATA_STR_TOKS_PARSED[seq][pos+1]!r} (expected {', '.join(list(map(repr, expected)))})"
        logits_topk = logits[seq, pos].topk(k, dim=-1, largest=not(neg))
        x = list(map(repr, model.to_str_tokens(logits_topk.indices)))
        y: list = logits_topk.values.tolist()
        color = ["#1F77B4"] * k

        # If the expected token is actually in the top k, then move it in there
        for str_tok_to_include in expected:
            tok_to_include = model.to_single_token(str_tok_to_include)
            for i, str_tok in enumerate(x):
                if repr(str_tok_to_include) == str_tok:
                    x[i] = x[i] + f" (#{i})"
                    color[i] = "#FF7F0E"
                    rank = i
                    break
            else:
                if neg: rank = (logits[seq, pos, tok_to_include] > logits[seq, pos]).sum().item()
                else: rank = (logits[seq, pos, tok_to_include] < logits[seq, pos]).sum().item()
                x.append(repr(str_tok_to_include)+ f" (#{rank})")
                y.append(logits[seq, pos, tok_to_include].item())
                color.append("#FF7F0E")

        # x = [f"{z} (#{i})" if not(z.endswith(")")) else z for i, z in enumerate(x)]

        px.bar(
            x=x, y=y, color=color, template="simple_white", title=f"({seq}, {pos}) {s}" if title is None else title,
            width=800, height=450, labels={"x": "Token", "y": "Logits", "color": "Token class"},
            color_discrete_map="identity"
        ).update_layout(
            xaxis_categoryorder = 'total ascending' if neg else 'total descending',
            hovermode="x unified", yaxis_range=[min(y) - 5, 0] if neg else [0, max(y) + 5], showlegend=False,
        ).show()



def plot_full_matrix_histogram(
    W_EE_dict: dict,
    src: Union[str, List[str]],
    dest: Union[str, List[str]],
    model: HookedTransformer,
    k: int = 10,
    head: Tuple[int, int] = (10, 7),
    neg: bool = True,
    circuit: Literal["QK", "OV"] = "OV",
    flip: bool = False,
):
    '''
    By default, this looks at what dest most attends to (QK) or what src causes to be most suppressed (OV).

    But if "flip" is True, then it looks at what things attend to src most (OV), or what causes dest to be most suppressed (OV).
    '''
    layer, head = head
    W_U: Tensor = W_EE_dict["W_U"].T
    W_O = model.W_O[layer, head]
    W_V = model.W_V[layer, head]
    W_Q = model.W_Q[layer, head]
    W_K = model.W_K[layer, head]
    W_OV = W_V @ W_O
    W_QK = W_Q @ W_K.T
    denom = (model.cfg.d_head ** 0.5)
    b_Q: Tensor = model.b_Q[layer, head]
    b_K: Tensor = model.b_K[layer, head]

    if isinstance(src, str): src = [src]
    if isinstance(dest, str): dest = [dest]
    src_toks = list(map(model.to_single_token, src))
    dest_toks = list(map(model.to_single_token, dest))

    W_EE = W_EE_dict["W_E (including MLPs)"]

    if circuit == "OV":
        if flip:
            assert len(dest_toks) == 1
            hist_toks = src_toks
            W_U_toks = W_U.T[dest_toks[0]]
            W_EE_scaled = W_EE / W_EE.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_EE_scaled @ W_OV @ W_U_toks
        else:
            assert len(src_toks) == 1
            hist_toks = dest_toks
            W_EE_OV_toks = W_EE[src_toks[0]] @ W_OV
            W_EE_OV_scaled_toks = W_EE_OV_toks / W_EE_OV_toks.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_EE_OV_scaled_toks @ W_U

        full_vector_topk = full_vector.topk(k, dim=-1, largest=not(neg))

    elif circuit == "QK":
        if flip:
            assert len(src_toks) == 1
            hist_toks = dest_toks
            W_EE_scaled_toks = W_EE[src_toks[0]] / W_EE[src_toks[0]].std(dim=-1, keepdim=True)
            W_U_scaled = W_U.T / W_U.T.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = W_U_scaled @ (W_QK @ W_EE_scaled_toks + W_Q @ b_K) / denom
        else:
            assert len(dest_toks) == 1
            hist_toks = src_toks
            W_U_scaled_toks = W_U.T[dest_toks[0]] / W_U.T[dest_toks[0]].std(dim=-1, keepdim=True)
            W_EE_scaled = W_EE / W_EE.std(dim=-1, keepdim=True)
            full_vector: Float[Tensor, "d_vocab"] = (W_U_scaled_toks @ W_QK + b_Q @ W_K.T) @ W_EE_scaled.T / denom

        full_vector_topk = full_vector.topk(k, dim=-1, largest=True)
    
    y = full_vector_topk.values.tolist()
    x = list(map(repr, model.to_str_tokens(full_vector_topk.indices)))
    color=["#1F77B4"] * k

    # If the expected token is actually in the top k, then move it in there
    for h_tok in hist_toks:
        h_str_tok = model.to_single_str_token(h_tok)
        for i, str_tok in enumerate(x):
            if str_tok == repr(h_str_tok):
                color[i] = "#FF7F0E"
                break
        else:
            if neg: rank = (full_vector[h_tok] > full_vector).sum().item()
            else: rank = (full_vector[h_tok] < full_vector).sum().item()
            x.append(repr(h_str_tok)+ f" (#{rank})")
            y.append(full_vector[h_tok].item())
            color.append("#FF7F0E")
    
    if circuit == "OV":
        if flip:
            title = f"<b style='font-size:22px;'>OV circuit</b>:<br>Which source tokens most suppress the prediction of<br><b>{dest[0]!r}</b> ?"
            x_label = "Source token"
        else:
            title = f"<b style='font-size:22px;'>OV circuit</b>:<br>Which predictions does source token <b>{src[0]!r}</b> suppress the most?"
            x_label = "Destination token (prediction)"
    else:
        if flip:
            title = f"<b style='font-size:22px;'>QK circuit</b>:<br>Which tokens' unembeddings most attend to source token <b>{src[0]!r}</b> ?"
            x_label = "Destination token (unembedding)"
        else:
            title = f"<b style='font-size:22px;'>QK circuit</b>:<br>Which source tokens does the unembedding of <b>{dest[0]!r}</b><br>attend to most?"
            x_label = "Source token"
    x_label = ""

    df = pd.DataFrame({
        "x": x, "y": y, "color": color
    })

    if neg:
        values_range=[min(y) - 10, 0]
        if max(y) > 0: values_range[1] = max(y) + 1
    else:
        values_range=[0, max(y) + 5]
        if min(y) < 0: values_range[0] = min(y) - 1

    fig = px.bar(
        df, x="y", y="x", color="color", template="simple_white", title=title,
        width=650, height=150+28*len(x), labels={"y": "Logits", "x": x_label, "color": "Token class"},
        color_discrete_map="identity", text_auto=".2f"
    ).update_layout(
        yaxis_categoryorder = 'total descending' if neg else 'total ascending',
        hovermode="y unified", xaxis_range=values_range, showlegend=False,
        margin_t=140, title_y=0.92,
    ).update_traces(
        textfont_size=13,
        textangle=0,
        textposition="outside",
        cliponaxis=False
    )
    fig.show()