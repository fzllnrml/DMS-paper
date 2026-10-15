"""Load-only checkpoint adapters. No training module is called."""
from __future__ import annotations

import numpy as np
from .distributions import Schedule


def choose_device(requested):
    import torch
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; use --device cpu")
    return requested


class LearnedField:
    kind = "neural"
    def __init__(self, checkpoint, config, mnist=False, device="cpu", batch_size=128):
        import torch
        self.device = choose_device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if mnist:
            from src.mnist.model import MNISTDiffusion
            self.model = MNISTDiffusion.from_checkpoint(checkpoint, self.device)
            actual = state["model_config"]
        else:
            from src.models.diffusion import DiffusionModel
            self.model = DiffusionModel(dim=int(state["dim"]), T=int(state["T"]),
                beta_start=config["beta_start"], beta_end=config["beta_end"],
                hidden_dims=state.get("config", {}).get("hidden_dims", [256, 256, 256]),
                activation=config["activation"], device=self.device)
            self.model.net.load_state_dict(state["state_dict"])
            self.model.net.eval()
            actual = dict(config, T=int(state["T"]))
        for k in ["T", "beta_start", "beta_end"]:
            if actual[k] != config[k]:
                raise ValueError(f"Checkpoint schedule {k} differs from experiment")
        self.model.net.requires_grad_(False)
        self.dim = self.model.dim
        self.schedule = Schedule(**{k: actual[k] for k in ["T", "beta_start", "beta_end"]})
        # Use the actual float32 schedule saved/reconstructed by the base model.
        self.schedule.alphas_cumprod = self.model.schedule.alphas_cumprod.detach().cpu().double().numpy()
        self.schedule.betas = self.model.schedule.betas.detach().cpu().double().numpy()
        self.batch_size, self.state, self.mnist = batch_size, state, mnist
        self.forward_calls = 0

    def score(self, x, t):
        import torch
        out = []
        with torch.no_grad():
            for begin in range(0, len(x), self.batch_size):
                a = torch.as_tensor(x[begin:begin + self.batch_size], dtype=torch.float32, device=self.device)
                out.append(self.model.score(a, t).cpu().numpy())
                self.forward_calls += 1
        return np.concatenate(out) if out else np.empty((0, self.dim))
