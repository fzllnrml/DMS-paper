"""Small offline correctness tests. No dataset downloads or full training."""

from copy import deepcopy
import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from src.mnist.data import load_mnist
from src.mnist.model import MNISTDiffusion
from src.mnist.sample import save_samples
from src.mnist.train import train, validation_loss


def fake_mnist(root, train, download):
    count = 16 if train else 5
    pixels = (torch.arange(count * 784).reshape(count, 28, 28) % 256).to(torch.uint8)
    # No targets or __getitem__: touching labels would fail these tests.
    return SimpleNamespace(data=pixels)


def tiny_model(device="cpu"):
    return MNISTDiffusion(T=12, base_channels=8, time_dim=16, device=device)


def tiny_config(output):
    return dict(
        seed=42, device="cpu", output_dir=str(output),
        data=dict(root="unused", download=False, validation_size=4, split_seed=11,
                  train_limit=8, validation_limit=4),
        model=dict(T=12, beta_start=0.0001, beta_end=0.02, base_channels=8, time_dim=16),
        training=dict(epochs=2, batch_size=4, lr=0.0002, weight_decay=0.,
                      grad_clip=1., num_workers=0, log_every=0),
    )


class MNISTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)

    @patch("src.mnist.data.MNIST", side_effect=fake_mnist)
    def test_disjoint_reproducible_splits_normalization_and_no_labels(self, _):
        splits = load_mnist(validation_size=4, split_seed=19)
        again = load_mnist(validation_size=4, split_seed=19)
        self.assertEqual((len(splits.train), len(splits.validation), len(splits.test)), (12, 4, 5))
        self.assertFalse(set(splits.train.indices.tolist()) & set(splits.validation.indices.tolist()))
        self.assertEqual(set(splits.train.indices.tolist()) | set(splits.validation.indices.tolist()), set(range(16)))
        torch.testing.assert_close(splits.train.indices, again.train.indices)
        self.assertIsNot(splits.train.pixels, splits.test.pixels)
        batch = next(iter(DataLoader(splits.train, batch_size=3)))
        self.assertEqual(batch.shape, (3, 1, 28, 28))
        self.assertEqual(batch.dtype, torch.float32)
        self.assertEqual(batch.min().item(), -1.)
        self.assertEqual(batch.max().item(), 1.)

    def test_forward_noising_and_batched_score_formula(self):
        model = tiny_model()
        x, noise = torch.randn(3, 1, 28, 28), torch.randn(3, 1, 28, 28)
        t = torch.tensor([0, 5, 11])
        a = model.schedule.alphas_cumprod[t, None, None, None]
        torch.testing.assert_close(model.q_sample(x, t, noise), a.sqrt() * x + (1 - a).sqrt() * noise)
        model.net.train()
        with torch.no_grad():
            expected = -model.net(x, t) / (1 - a).sqrt()
        model.enable_nfe_counting()
        actual = model.score(x, t)
        self.assertTrue(model.net.training)
        self.assertFalse(actual.requires_grad)
        self.assertEqual(model.nfe, 3)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.score(x.flatten(1), t), actual.flatten(1))
        for i in range(3):
            # CPU convolution rounding differs with batch size. Compare in
            # epsilon units so t=0's 1/sigma factor does not amplify the tolerance.
            sigma = model.schedule.sqrt_one_minus_alphas_cumprod[t[i]]
            torch.testing.assert_close(model.score(x[i].flatten(), int(t[i])) * sigma,
                                       actual[i].flatten() * sigma, rtol=2e-5, atol=2e-6)
        self.assertEqual(model.score_numpy(x.flatten(1).numpy(), t).shape, (3, 784))
        self.assertTrue(torch.isfinite(actual).all())

    def test_invalid_score_inputs(self):
        model = tiny_model()
        x = torch.zeros(2, 784)
        for t in (-1, 12, 0.5, True, torch.tensor([0]), torch.tensor([[0], [1]])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                model.score(x, t)
        with self.assertRaises(ValueError):
            model.score(torch.zeros(2, 28, 28), 0)
        with self.assertRaises(ValueError):
            model.score(x.double(), 0)

    def test_ddim_visits_zero_and_finishes_at_clean_endpoint(self):
        model = tiny_model()
        seen = []

        def oracle(x, t):
            seen.append(int(t[0]))
            a = model.schedule.alphas_cumprod[t, None, None, None]
            return (x - a.sqrt() * 0.25) / (1 - a).sqrt()

        with patch.object(model.net, "forward", side_effect=oracle):
            model.enable_nfe_counting()
            samples = model.sample_ddim(n=2, num_steps=4, clip_denoised=False)
        self.assertEqual(seen, [11, 7, 4, 0])
        self.assertEqual(model.nfe, 8)
        torch.testing.assert_close(samples, torch.full_like(samples, 0.25), atol=1e-6, rtol=1e-6)

    def test_ddim_seed_rng_isolation_png_and_bounds(self):
        model = tiny_model()
        state = torch.get_rng_state().clone()
        samples = model.sample_ddim(n=2, num_steps=3, seed=17)
        torch.testing.assert_close(torch.get_rng_state(), state)
        torch.testing.assert_close(samples, model.sample_ddim(n=2, num_steps=3, seed=17), rtol=0, atol=0)
        self.assertFalse(torch.equal(samples, model.sample_ddim(n=2, num_steps=3, seed=18)))
        self.assertTrue(torch.isfinite(samples).all())
        self.assertLessEqual(samples.abs().max().item(), 1.)
        with TemporaryDirectory() as directory:
            path = save_samples(model, Path(directory) / "grid.png", n=2, steps=2)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for steps in (0, 13):
            with self.assertRaises(ValueError):
                model.sample_ddim(n=2, num_steps=steps)

    @patch("src.mnist.data.MNIST", side_effect=fake_mnist)
    def test_fixed_validation_does_not_change_weights_or_mode(self, _):
        model = tiny_model()
        loader = DataLoader(load_mnist(validation_size=4).validation, batch_size=2,
                            generator=torch.Generator().manual_seed(1))
        before = {k: v.clone() for k, v in model.net.state_dict().items()}
        model.net.train()
        first = validation_loss(model, loader, seed=7)
        self.assertEqual(first, validation_loss(model, loader, seed=7))
        self.assertTrue(model.net.training)
        for key, value in before.items():
            torch.testing.assert_close(value, model.net.state_dict()[key], rtol=0, atol=0)

    @patch("src.mnist.data.MNIST", side_effect=fake_mnist)
    def test_checkpoint_resume_matches_uninterrupted_training(self, _):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            full_path = train(tiny_config(root / "full"))
            config = tiny_config(root / "resumed")
            config["training"]["epochs"] = 1
            resumed_path = train(config)
            first = torch.load(resumed_path, weights_only=True)
            self.assertEqual(first["global_step"], 2)
            config["training"]["epochs"] = 2
            train(config, resume=resumed_path)
            full = torch.load(full_path, weights_only=True)
            resumed = torch.load(resumed_path, weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["global_step"], 4)
            self.assertEqual(full["metrics"], resumed["metrics"])
            for key in full["model"]:
                torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
            for key in ("order_rng", "noise_rng"):
                torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)
            for parameter, state in full["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, resumed["optimizer"]["state"][parameter][key], rtol=0, atol=0)
            loaded = MNISTDiffusion.from_checkpoint(resumed_path, "cpu")
            self.assertTrue(torch.isfinite(loaded.score(torch.zeros(2, 784), torch.tensor([0, 11]))).all())
            self.assertTrue((resumed_path.parent / "best.pt").exists())
            self.assertEqual(len((root / "resumed/logs/metrics.jsonl").read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                train(config)
            bad = deepcopy(config)
            bad["training"]["epochs"] = 3
            bad["data"]["split_seed"] = 12
            with self.assertRaisesRegex(ValueError, "Resume config differs"):
                train(bad, resume=resumed_path)

    def check_backend(self, device):
        model = tiny_model(device)
        x = torch.randn(2, 1, 28, 28).to(model.device)
        t = torch.tensor([0, 11], device=model.device)
        noise = torch.randn_like(x)
        xt = model.q_sample(x, t, noise)
        loss = (model.net(xt, t) - noise).square().mean()
        loss.backward()
        torch.optim.AdamW(model.net.parameters()).step()
        self.assertTrue(torch.isfinite(model.score(x, t)).all())
        self.assertTrue(torch.isfinite(model.sample_ddim(n=2, num_steps=2)).all())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS hardware unavailable")
    def test_mps(self):
        self.check_backend("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
    def test_cuda(self):
        self.check_backend("cuda")


if __name__ == "__main__":
    unittest.main()
