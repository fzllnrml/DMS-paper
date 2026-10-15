"""Generate a PNG grid with deterministic DDIM from a saved checkpoint."""

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from .model import MNISTDiffusion


def save_samples(model, output, n=64, steps=50, seed=42):
    samples = model.sample_ddim(n=n, num_steps=steps, seed=seed)
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Nonfinite DDIM samples")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image((samples.cpu() + 1) / 2, output, nrow=math.ceil(math.sqrt(n)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output", default="runs/mnist/samples/ddim.png")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = MNISTDiffusion.from_checkpoint(args.checkpoint, args.device)
    print(save_samples(model, args.output, n=args.n, steps=args.steps, seed=args.seed))


if __name__ == "__main__":
    main()
