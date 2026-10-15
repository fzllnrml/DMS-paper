"""Train with epoch-boundary checkpoints and reproducible validation noise."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from .data import load_mnist
from .model import MNISTDiffusion


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Unsupported deterministic kernels emit a warning, including on MPS.
    torch.use_deterministic_algorithms(True, warn_only=True)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def noisy_batch(model, images, generator):
    # Explicit CPU generator: supported by every backend and saved for resume.
    noise = torch.randn(images.shape, generator=generator, dtype=torch.float32).to(model.device)
    t = torch.randint(model.schedule.T, (len(images),), generator=generator).to(model.device)
    return model.q_sample(images, t, noise), t, noise


def validation_loss(model, loader, seed):
    generator = torch.Generator().manual_seed(seed)
    total, count = 0., 0
    with model._evaluation():
        for images in loader:
            images = images.to(model.device)
            xt, t, noise = noisy_batch(model, images, generator)
            loss = F.mse_loss(model.net(xt, t), noise)
            total += loss.item() * len(images)
            count += len(images)
    if count == 0:
        raise ValueError("Empty validation loader")
    return total / count


def resume_signature(config):
    """Only runtime paths, device, logging frequency and total epochs may change."""
    signature = deepcopy(config)
    for key in ("device", "output_dir"):
        signature.pop(key, None)
    for key in ("root", "download"):
        signature["data"].pop(key, None)
    for key in ("epochs", "log_every"):
        signature["training"].pop(key, None)
    return signature


def train(config, resume=None):
    config = deepcopy(config)
    settings = config["training"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["num_workers"] < 0:
        raise ValueError("epochs/batch_size must be positive; num_workers must be nonnegative")
    if settings["lr"] <= 0 or settings["grad_clip"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Require lr/grad_clip > 0 and weight_decay >= 0")
    output = Path(config["output_dir"])
    latest = output / "checkpoints/latest.pt"
    if resume is None and (latest.exists() or (output / "logs/metrics.jsonl").exists()):
        raise FileExistsError(f"Run already exists: {output}. Use --resume or a new --output-dir")
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "mnist_diffusion_v1":
            raise ValueError("Not an MNIST diffusion checkpoint")
        if resume_signature(checkpoint["config"]) != resume_signature(config):
            raise ValueError("Resume config differs from checkpoint (model/data/seed/training settings)")
        if checkpoint["epoch"] >= settings["epochs"]:
            raise ValueError("--epochs is the total target; it must exceed the checkpoint epoch")
        if latest.exists() and Path(resume).resolve() != latest.resolve():
            raise ValueError("Use --output-dir with a new directory when resuming a different checkpoint")

    seed_everything(config["seed"])
    model = MNISTDiffusion(**config["model"], device=config["device"])
    splits = load_mnist(**config["data"])
    split_indices = {"train": splits.train.indices, "validation": splits.validation.indices,
                     "test_size": len(splits.test)}
    order_rng = torch.Generator().manual_seed(config["seed"] + 1)
    noise_rng = torch.Generator().manual_seed(config["seed"] + 2)
    train_loader = DataLoader(
        splits.train, batch_size=settings["batch_size"], shuffle=True,
        generator=order_rng, num_workers=settings["num_workers"],
        pin_memory=model.device.type == "cuda",
    )
    val_loader = DataLoader(
        splits.validation, batch_size=settings["batch_size"], shuffle=False,
        num_workers=settings["num_workers"],
        generator=torch.Generator().manual_seed(config["seed"] + 3),
    )
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=settings["lr"],
                                 weight_decay=settings["weight_decay"])
    start_epoch, global_step, best_loss = 0, 0, math.inf
    if checkpoint:
        saved_splits = checkpoint["split_indices"]
        if (not torch.equal(saved_splits["train"], split_indices["train"])
                or not torch.equal(saved_splits["validation"], split_indices["validation"])
                or saved_splits["test_size"] != split_indices["test_size"]):
            raise ValueError("MNIST split does not match checkpoint")
        model.net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
        best_loss = checkpoint["best_validation_loss"]
        order_rng.set_state(checkpoint["order_rng"])
        noise_rng.set_state(checkpoint["noise_rng"])
        restore_rng(checkpoint["rng"])

    (output / "logs").mkdir(parents=True, exist_ok=True)
    with (output / "config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    metrics_path = output / "logs/metrics.jsonl"
    # A crash after logging but before checkpointing must not duplicate epochs.
    if resume and metrics_path.exists():
        records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        metrics_path.write_text("".join(json.dumps(row) + "\n" for row in records if row["epoch"] <= start_epoch))
    print(f"device={model.device} parameters={sum(p.numel() for p in model.net.parameters()):,} "
          f"train={len(splits.train)} validation={len(splits.validation)} test={len(splits.test)}", flush=True)

    for epoch in range(start_epoch + 1, settings["epochs"] + 1):
        model.net.train()
        total, count = 0., 0
        for images in train_loader:
            images = images.to(model.device)
            xt, t, noise = noisy_batch(model, images, noise_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.net(xt, t), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; last completed epoch checkpoint is intact")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            total += loss.item() * len(images)
            count += len(images)
            if settings["log_every"] and global_step % settings["log_every"] == 0:
                print(f"epoch={epoch} step={global_step} batch_loss={loss.item():.6f}", flush=True)
        val_loss = validation_loss(model, val_loader, config["seed"] + 4)
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        row = dict(epoch=epoch, global_step=global_step, train_loss=total / count,
                   validation_loss=val_loss, lr=optimizer.param_groups[0]["lr"])
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        state = dict(
            format="mnist_diffusion_v1", config=config, model_config=model.config,
            model=model.net.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, global_step=global_step, best_validation_loss=best_loss,
            metrics=row, split_indices=split_indices, order_rng=order_rng.get_state(),
            noise_rng=noise_rng.get_state(), rng=rng_state(),
            torch_version=str(torch.__version__), device=str(model.device),
        )
        # Each file is atomically replaced. Resume always uses a complete epoch.
        atomic_save(state, latest)
        if improved:
            atomic_save(state, output / "checkpoints/best.pt")
        print(json.dumps(row), flush=True)
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, help="Total epochs, including completed epochs on resume")
    parser.add_argument("--resume", help="latest.pt or best.pt")
    parser.add_argument("--smoke", action="store_true", help="Only 32 train / 16 validation images, one epoch by default")
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    if args.smoke:
        config["data"].update(train_limit=32, validation_limit=16)
        config["training"].update(batch_size=16, epochs=1)
        config["output_dir"] = "runs/mnist_smoke"
    if args.device:
        config["device"] = args.device
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    checkpoint = train(config, resume=args.resume)
    if args.smoke:
        from .sample import save_samples
        model = MNISTDiffusion.from_checkpoint(checkpoint, config["device"])
        save_samples(model, Path(config["output_dir"]) / "samples/smoke.png", n=4, steps=4, seed=config["seed"])
        images = torch.zeros(2, 1, 28, 28, device=model.device)
        scores = model.score(images.flatten(1), torch.tensor([0, model.schedule.T - 1]))
        if scores.shape != (2, 784) or not torch.isfinite(scores).all():
            raise RuntimeError("Score smoke check failed")
        print("Smoke passed: training, validation, checkpoint reload, DDIM PNG, batched score", flush=True)


if __name__ == "__main__":
    main()
