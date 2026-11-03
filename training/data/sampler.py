"""Reproducible source-weighted sampling without a dataset-sized index list."""

from __future__ import annotations

from collections.abc import Iterator
from operator import index

import numpy as np
from torch.utils.data import Sampler


class WeightedSourceSampler(Sampler[int]):
    """Draw a weighted source, then a uniform row; each iterator resets its seed."""

    def __init__(
        self, source_ranges: list[tuple[int, int]], weights: list[float],
        num_samples: int, seed: int,
    ):
        self.num_samples = index(num_samples)
        self.seed = index(seed)
        if self.num_samples < 0 or self.seed < 0:
            raise ValueError("num_samples and seed must be nonnegative")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(source_ranges):
            raise ValueError("one weight is required per source range")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and nonnegative")
        ranges, active_weights = [], []
        for (start, end), weight in zip(source_ranges, weights):
            start, end = index(start), index(end)
            if not 0 <= start <= end <= np.iinfo(np.int64).max:
                raise ValueError("source ranges must satisfy 0 <= start <= end <= int64 max")
            if start < end and weight > 0:
                ranges.append((start, end))
                active_weights.append(weight)
        if self.num_samples and not ranges:
            raise ValueError("sampling requires a nonempty source with positive weight")
        self._ranges = ranges
        self._weights = np.asarray(active_weights, dtype=np.float64)
        if ranges:
            # Scale first so even very large finite weights normalize safely.
            self._weights /= self._weights.max()
            self._weights /= self._weights.sum()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            source = int(rng.choice(len(self._ranges), p=self._weights))
            start, end = self._ranges[source]
            yield int(rng.integers(start, end))
