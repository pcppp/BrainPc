r"""Episodic meta-learning utilities for site-level domain generalization.

Provides:
  - build_site_loaders: group a training dataset by data.env_id and build
    one PyG DataLoader per site (with shuffle and per-site cyclic sampling).
  - sample_episode: choose one source site as meta-test, the rest as meta-train.
  - next_site_batch: fetch next batch from a site's iterator, auto-resetting
    when the iterator is exhausted (cyclic sampling for small sites).

These helpers do NOT modify the model forward pass or the loss formulation.
They only reorganise how training batches are drawn within each step.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def _env_id_value(data) -> int:
    """Return per-graph env_id as a Python int."""
    env = getattr(data, 'env_id', None)
    if env is None:
        env = getattr(data, 'domain', None)
    if env is None:
        raise ValueError('Sample is missing env_id / domain attribute')
    if torch.is_tensor(env):
        return int(env.view(-1)[0].item())
    return int(env)


def build_site_loaders(
    train_dataset,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> Tuple[Dict[int, DataLoader], List[int], Dict[int, int]]:
    """Group train_dataset by env_id and build one DataLoader per site.

    Args:
        train_dataset: iterable of PyG Data objects (each must carry env_id).
        batch_size:    per-site batch size.
        num_workers:   DataLoader workers.
        seed:          base seed for per-site torch.Generator (for reproducibility).

    Returns:
        site_loaders: {site_id -> DataLoader}
        source_sites: sorted list of unique site ids
        site_sizes:   {site_id -> #samples}
    """
    by_site: Dict[int, list] = {}
    for sample in train_dataset:
        sid = _env_id_value(sample)
        by_site.setdefault(sid, []).append(sample)

    source_sites = sorted(by_site.keys())
    site_sizes = {s: len(by_site[s]) for s in source_sites}

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    site_loaders: Dict[int, DataLoader] = {}
    for sid in source_sites:
        sub = by_site[sid]
        # If site has fewer samples than batch_size, drop_last=False so we still
        # yield at least one batch per site; cyclic sampling in next_site_batch
        # will refill on exhaustion.
        bs = min(batch_size, max(1, len(sub)))
        g = torch.Generator()
        g.manual_seed(seed + int(sid))
        site_loaders[sid] = DataLoader(
            sub,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            generator=g,
            drop_last=False,
        )
    return site_loaders, source_sites, site_sizes


def sample_episode(
    source_sites: List[int],
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], int]:
    """Pick one source site as meta_test, the rest as meta_train.

    Only training source sites are used — never the real OOD test site —
    so this cannot leak unseen-domain data into training.
    """
    if len(source_sites) < 2:
        raise ValueError(f'Need >=2 source sites for an episode, got {source_sites}')
    pick = (rng.choice if rng is not None else random.choice)(source_sites)
    meta_train_sites = [s for s in source_sites if s != pick]
    return meta_train_sites, pick


def next_site_batch(
    site_iterators: Dict[int, Iterable],
    site_loaders: Dict[int, DataLoader],
    site: int,
):
    """Get next batch for a site, auto-resetting iterator on exhaustion.

    This is the mechanism that lets small sites keep producing batches every
    episode even when the per-site epoch is much shorter than the global one.
    """
    try:
        return next(site_iterators[site])
    except (StopIteration, KeyError):
        site_iterators[site] = iter(site_loaders[site])
        return next(site_iterators[site])
