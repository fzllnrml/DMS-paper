"""Official MNIST test set stays separate; labels never enter a batch."""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class ImageOnlyDataset(Dataset):
    def __init__(self, pixels, indices):
        self.pixels = pixels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # uint8 -> float32 [-1,1], no fitted statistics or stochastic transforms.
        return self.pixels[self.indices[index]].unsqueeze(0).float().div(127.5).sub(1)


@dataclass
class MNISTSplits:
    train: ImageOnlyDataset
    validation: ImageOnlyDataset
    test: ImageOnlyDataset


def load_mnist(root="data/mnist", validation_size=5000, split_seed=42,
               download=True, train_limit=None, validation_limit=None):
    train = MNIST(root=root, train=True, download=download)
    test = MNIST(root=root, train=False, download=download)
    if not 0 < validation_size < len(train.data):
        raise ValueError("validation_size must leave nonempty train and validation sets")
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(train.data), generator=generator)
    val_indices, train_indices = indices[:validation_size], indices[validation_size:]

    def limited(indices, limit):
        if limit is not None and not 0 < limit <= len(indices):
            raise ValueError("Subset limit must be positive and no greater than split size")
        return indices if limit is None else indices[:limit]

    return MNISTSplits(
        ImageOnlyDataset(train.data, limited(train_indices, train_limit)),
        ImageOnlyDataset(train.data, limited(val_indices, validation_limit)),
        ImageOnlyDataset(test.data, torch.arange(len(test.data))),
    )
