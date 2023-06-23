from transformer_lens.cautils.utils import *

def print_best_outputs(
    best_k_indices: List[Tuple[int, int]], # list of (batch_idx, seq_pos) tuples
    best_k_loss_decrease: List[Float], # sorted loss increases from ablating 10.7
    hook: Tuple[str, Callable], # (hook_name, hook_fn) for doing ablation
    model: HookedTransformer,
    data: List[str],
    n: int = 10,
    random: bool = False,
    window: int = 200
):
    assert len(best_k_indices) == len(best_k_loss_decrease)
    assert sorted(best_k_loss_decrease, reverse=True) == best_k_loss_decrease

    if random:
        indices_to_print = t.randperm(len(best_k_indices))[:n].tolist()
    else:
        indices_to_print = list(range(n))

    for i in indices_to_print:

        # Get the indices of the token where loss was calculated from
        sample_idx, token_idx = best_k_indices[i]
        loss_decrease = best_k_loss_decrease[i]

        # Get the string leading up to that point, and a few other useful things e.g. the correct next token
        new_str = data[sample_idx]
        new_str_tokens = model.to_str_tokens(new_str)
        tokens = model.to_tokens(new_str)[:, :token_idx+1]
        prev_tokens = "".join(new_str_tokens[max(0, token_idx - window): token_idx + 1])
        next_tokens = "".join(new_str_tokens[token_idx + 2: min(token_idx + 10, len(new_str_tokens))])
        correct_str_token = new_str_tokens[token_idx + 1]
        correct_token = model.to_tokens(new_str)[0, token_idx+1]

        # Get logits and stuff for the original (non-ablated) distribution
        logits = model(tokens, return_type="logits")[0, -1]
        log_probs = logits.log_softmax(-1)
        sorted_log_probs = t.sort(log_probs, descending=False).values
        top_word_posn = (model.cfg.d_vocab - t.searchsorted(sorted_log_probs, log_probs[correct_token].item())).item()
        topk = log_probs.topk(5, dim=-1)
        top_logits = t.concat([logits[topk.indices], logits[correct_token].unsqueeze(0)])
        top_word_logprobs, top_word_indices = topk
        top_word_logprobs = t.concat([top_word_logprobs.squeeze(), log_probs[correct_token].unsqueeze(0)])
        top_words = model.to_str_tokens(top_word_indices.squeeze()) + [correct_str_token]
        rprint_output = [f"[dark_orange bold]logit = {logit:.3f}[/] | [bright_red bold]prob = {logprob.exp():.3f}[/] | {repr(word)}" for logit, logprob, word in zip(top_logits, top_word_logprobs, top_words)]
        
        # Get logits and stuff for the original ablated distribution
        ablated_logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=[hook])[0, -1]
        ablated_log_probs = ablated_logits.log_softmax(-1)
        sorted_log_probs = t.sort(ablated_log_probs, descending=False).values
        top_word_posn_ablated = (model.cfg.d_vocab - t.searchsorted(sorted_log_probs, ablated_log_probs[correct_token].item())).item()
        topk = ablated_log_probs.topk(5, dim=-1)
        top_logits = t.concat([ablated_logits[topk.indices], ablated_logits[correct_token].unsqueeze(0)])
        top_word_logprobs, top_word_indices = topk
        top_word_logprobs = t.concat([top_word_logprobs.squeeze(), ablated_log_probs[correct_token].unsqueeze(0)])
        top_words = model.to_str_tokens(top_word_indices.squeeze()) + [correct_str_token]
        rprint_output_ablated = [f"[dark_orange bold]logit = {logit:.3f}[/] | [bright_red bold]prob = {logprob.exp():.3f}[/] | {repr(word)}" for logit, logprob, word in zip(top_logits, top_word_logprobs, top_words)]

        # Create and display table
        table = Table("Original", "Ablated", title=f"Correct = {repr(correct_str_token)}, Loss decrease from NNMH = {loss_decrease:.3f}")
        table.add_row("Top words:", "Top words:")
        table.add_row("", "")
        for output, output_ablated in zip(rprint_output[:-1], rprint_output_ablated):
            table.add_row(output, output_ablated)
        table.add_row("", "")
        table.add_row(f"Correct word (predicted at posn {top_word_posn}):", f"Correct word (predicted at posn {top_word_posn_ablated}):")
        table.add_row("", "")
        table.add_row(rprint_output[-1], rprint_output_ablated[-1])

        rprint(prev_tokens.replace("\n", "") + f"[dark_orange bold u]{correct_str_token}[/]" + next_tokens.replace("\n", ""))
        rprint(table)