from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class GenerationCompatibilityTests(unittest.TestCase):
    def test_no_restep_generate_samples_from_cached_logits_first(self) -> None:
        import torch

        from rnn.model import ModelCfg, SettleCharLM

        torch.manual_seed(123)
        model = SettleCharLM(
            5, ModelCfg(d_model=8, n_layers=1, k_settle=1, detach_state=True)
        )
        model.reset_state(batch_size=1)
        prompt_token = torch.tensor([2], dtype=torch.long)
        model.step(prompt_token)
        cached_argmax = int(torch.argmax(model.last_logits, dim=-1).item())

        stepped_tokens: list[int] = []
        original_step = model.step

        def counting_step(token_id, **kwargs):
            stepped_tokens.append(int(token_id.item()))
            return original_step(token_id, **kwargs)

        model.step = counting_step  # type: ignore[method-assign]
        generated, _ = model.generate(1, temperature=0.0, restep_last_token=False)

        self.assertEqual(int(generated[0, 0].item()), cached_argmax)
        self.assertEqual(stepped_tokens, [cached_argmax])

    def test_legacy_restep_generate_steps_last_prompt_token_first(self) -> None:
        import torch

        from rnn.model import ModelCfg, SettleCharLM

        torch.manual_seed(123)
        model = SettleCharLM(
            5, ModelCfg(d_model=8, n_layers=1, k_settle=1, detach_state=True)
        )
        model.reset_state(batch_size=1)
        prompt_token = torch.tensor([2], dtype=torch.long)
        model.step(prompt_token)

        stepped_tokens: list[int] = []
        original_step = model.step

        def counting_step(token_id, **kwargs):
            stepped_tokens.append(int(token_id.item()))
            return original_step(token_id, **kwargs)

        model.step = counting_step  # type: ignore[method-assign]
        model.generate(1, temperature=0.0, restep_last_token=True)

        self.assertEqual(stepped_tokens, [2])


if __name__ == "__main__":
    unittest.main()
