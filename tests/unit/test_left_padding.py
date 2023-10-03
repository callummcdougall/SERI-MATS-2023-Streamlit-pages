import pytest
import torch

from transformer_lens import HookedTransformer, utils


class TestLeftPadding:
    prompts = [
        "Hello world!",
        "How are you today?",
        "I'm fine, thank you.",
        "I am happy.",
    ]

    # helpers
    def check_outputs_identity(
        self,
        i,
        single_outputs,
        left_outputs,
        right_outputs,
        left_token_start,
        left_token_end,
        right_token_start,
        right_token_end,
    ):
        atol = 1e-4

        assert torch.allclose(
            left_outputs[i, left_token_start:left_token_end, :],
            right_outputs[i, right_token_start:right_token_end, :],
            atol=atol,
        )

        assert torch.allclose(
            left_outputs[i, left_token_start:left_token_end, :],
            single_outputs[0],
            atol=atol,
        )

        assert torch.allclose(
            right_outputs[i, right_token_start:right_token_end, :],
            single_outputs[0],
            atol=atol,
        )

    # fixtures
    @pytest.fixture(scope="class", params=["gpt2-small", "facebook/opt-125m"])
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        model = HookedTransformer.from_pretrained(model_name)
        return model

    # tests
    @pytest.mark.parametrize("padding_side", ["left", "right"])
    @pytest.mark.parametrize("prepend_bos", [True, False])
    def test_pos_embed(self, model, padding_side, prepend_bos):
        # setup
        model.tokenizer.padding_side = padding_side

        prompts = self.prompts
        tokens = model.to_tokens(prompts, prepend_bos=prepend_bos)
        str_tokens = model.to_str_tokens(prompts, prepend_bos=prepend_bos)

        left_attention_mask = utils.get_attention_mask(
            model.tokenizer, tokens, prepend_bos
        )  # [batch pos]

        output_pos_embed = model.pos_embed(
            tokens, 0, left_attention_mask=left_attention_mask
        )  # [batch pos d_model]

        # check if the output pos_embeds have the correct shape
        assert output_pos_embed.shape == (
            tokens.shape[0],
            tokens.shape[1],
            model.pos_embed.W_pos.shape[1],
        )

        # check if the target pos_embeds are the same as the output pos_embeds
        target_position_ids = torch.tensor(
            sum([list(range(len(t))) for t in str_tokens], []), device=tokens.device
        )
        target_output_pos_embed = model.pos_embed.W_pos[target_position_ids, :]

        attended_output_pos_embed = output_pos_embed[left_attention_mask.bool()]

        assert torch.allclose(
            attended_output_pos_embed, target_output_pos_embed, atol=1e-4
        )

        # padded positions should have zero pos_embed
        assert output_pos_embed[~left_attention_mask.bool()].sum() == 0

    def test_left_padding_by_comparing_outputs(self, model):
        prompts = self.prompts

        num_str_tokens_list = [len(t) for t in model.to_str_tokens(prompts)]

        # left padding output
        model.tokenizer.padding_side = "left"
        left_logits, left_cache = model.run_with_cache(prompts)
        left_last_logits = left_logits[:, -1, :]
        left_first_token_positions = left_logits.shape[1] - torch.tensor(
            num_str_tokens_list, device=left_logits.device
        )
        left_first_logits = left_logits[
            torch.arange(len(prompts)), left_first_token_positions, :
        ].squeeze(1)

        # right padding output
        model.tokenizer.padding_side = "right"
        right_logits, right_cache = model.run_with_cache(prompts)
        right_last_token_positions = (
            torch.tensor(num_str_tokens_list, device=right_logits.device) - 1
        )
        right_last_logits = right_logits[
            torch.arange(len(prompts)), right_last_token_positions, :
        ].squeeze(1)
        right_first_logits = right_logits[:, 0, :]

        # check if the left and right padding outputs are the same for the first and last tokens
        assert torch.allclose(left_last_logits, right_last_logits, atol=1e-4)
        assert torch.allclose(left_first_logits, right_first_logits, atol=1e-4)

        # check if the left and right padding outputs are the same for all tokens
        # and if the batched padded outputs are the same as the single prompt outputs
        right_token_start = 0
        left_token_end = left_logits.shape[1]
        for i, (prompt, left_token_start, right_token_end) in enumerate(
            zip(
                prompts,
                left_first_token_positions.tolist(),
                (right_last_token_positions + 1).tolist(),
            )
        ):
            single_logits, single_cache = model.run_with_cache(prompt)

            assert (
                right_token_end - right_token_start
                == left_token_end - left_token_start
                == single_logits.shape[1]
            )

            self.check_outputs_identity(
                i,
                single_logits,
                left_logits,
                right_logits,
                left_token_start,
                left_token_end,
                right_token_start,
                right_token_end,
            )

            # check cache
            for name in ["k6a", "pre2", "embed", "k6", "scale4ln1", "pre5"]:
                self.check_outputs_identity(
                    i,
                    single_cache[name],
                    left_cache[name],
                    right_cache[name],
                    left_token_start,
                    left_token_end,
                    right_token_start,
                    right_token_end,
                )
