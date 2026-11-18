"""Ultralytics trainer for the unified detector.

Three changes relative to the stock ``DetectionTrainer``:

1. builds ``UnifiedDetectionModel`` and transfers pretrained YOLO weights into it;
2. draws training images with equal probability from each sensor, so the SAR set
   (about 10x larger than the others) cannot dominate;
3. keeps mosaic augmentation inside one sensor. Stock mosaic stitches four random
   images together, which here would paste a SAR tile next to an infrared one
   and make the modality of the resulting image meaningless.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.augment import Mosaic
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from .model import UnifiedDetectionModel
from .modality import MODALITIES, modality_of


class ModalityMosaic(Mosaic):
    """Mosaic that only ever tiles images from the same sensor as the current one."""

    def get_indexes(self):
        pool = self.dataset.indices_by_modality[self.dataset.current_modality]
        if self.buffer_enabled:
            buffered = [i for i in self.dataset.buffer if self.dataset.modality_ids[i] == self.dataset.current_modality]
            if len(buffered) >= self.n - 1:
                pool = buffered
        return [random.choice(pool) for _ in range(self.n - 1)]


class ModalityYOLODataset(YOLODataset):
    """YOLODataset that knows each image's sensor."""

    def init_modality(self):
        self.modality_ids = np.array([modality_of(p) for p in self.im_files])
        self.indices_by_modality = {
            m: np.flatnonzero(self.modality_ids == m).tolist() for m in range(len(MODALITIES))}
        self.current_modality = 0
        self._patch_mosaic()

    def _patch_mosaic(self):
        for t in getattr(self.transforms, "transforms", []):
            if type(t) is Mosaic:
                t.__class__ = ModalityMosaic

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        self.transforms = transforms
        if hasattr(self, "modality_ids"):
            self._patch_mosaic()
        return transforms

    def __getitem__(self, index):
        self.current_modality = int(self.modality_ids[index])
        return super().__getitem__(index)


class UnifiedTrainer(DetectionTrainer):
    """Set the class attributes below before instantiating."""

    model_kwargs: dict = {}
    epoch_size: int | None = None  # images drawn per epoch; None = size of the pooled train set
    balanced: bool = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = UnifiedDetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"],
            verbose=verbose and RANK == -1, **self.model_kwargs)
        model = self.set_model_names_for_load(model)
        if weights is not None:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.__class__ = ModalityYOLODataset
        dataset.init_modality()
        return dataset

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train" or not self.balanced:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        counts = np.bincount(dataset.modality_ids, minlength=len(MODALITIES))
        weights = 1.0 / np.maximum(counts, 1)[dataset.modality_ids]
        n = self.epoch_size or len(dataset)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), n, replacement=True)
        LOGGER.info(
            "Balanced sampling: " + ", ".join(f"{m}={c}" for m, c in zip(MODALITIES, counts))
            + f" images -> ~{n // len(MODALITIES)} drawn per sensor per epoch")

        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        batch_size = min(batch_size, n)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=min(os.cpu_count() or 1, self.args.workers),
            prefetch_factor=4 if self.args.workers > 0 else None,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        )
