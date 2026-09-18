"""CPU model regression tests; no audio downloads or trained weights."""
import contextlib
import io
import unittest
import torch
from token_unet import TokenUNet
from token_constants import PAD


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def model(self, **kwargs):
        return TokenUNet(n_q=2, k=8, base_dim=8, depth=2, dropout=0., **kwargs)

    def test_short_odd_and_aligned_lengths(self):
        model = self.model().eval()
        for length in [1, 2, 3, 4, 15, 16, 17, 31, 32]:
            output = model(torch.randint(8, (2, 2, length)))
            self.assertEqual(output['logits'].shape, (2, 8, 2, length))
            self.assertEqual(output['mask'].shape, (2, 2, length))
            self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))

    def test_padding_and_quiet_default(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            output = self.model()(torch.full((2, 2, 7), PAD))
        self.assertEqual(stream.getvalue(), '')
        self.assertTrue(torch.isfinite(output['logits']).all())

    def test_checkpointed_gradients_match(self):
        plain, checked = self.model(), self.model(checkpointing=True)
        checked.load_state_dict(plain.state_dict())
        tokens = torch.randint(8, (2, 2, 17))
        for model in (plain, checked):
            sum(value.square().mean() for value in model(tokens).values()).backward()
        for p, q in zip(plain.parameters(), checked.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertIsNotNone(q.grad)
            torch.testing.assert_close(p.grad, q.grad)

    def test_invalid_tokens(self):
        model = self.model()
        for tokens in [torch.zeros(2, 2, 4), torch.zeros(2, 3, 4, dtype=torch.long),
                       torch.full((2, 2, 4), 8), torch.full((2, 2, 4), -1),
                       torch.zeros(2, 2, 0, dtype=torch.long)]:
            with self.assertRaises(ValueError):
                model(tokens)

    def test_invalid_configuration(self):
        for config in [{'n_q': 0}, {'k': 0}, {'base_dim': 7}, {'depth': 0}, {'dropout': 1.}]:
            args = dict(n_q=2, k=8, base_dim=8, depth=2, dropout=0.)
            args.update(config)
            with self.assertRaises(ValueError):
                TokenUNet(**args)
        with self.assertRaises(ValueError):
            self.model().set_dropout(-.1)

    def test_optional_skip_preserves_state_dict_contract(self):
        original = self.model()
        variant = self.model(full_resolution_skip=True)
        variant.load_state_dict(original.state_dict(), strict=True)
        tokens = torch.randint(8, (2, 2, 17))
        loss = variant(tokens)['logits'].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(variant.emb.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
