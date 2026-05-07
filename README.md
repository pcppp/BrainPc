# BrainOOD

This is the official PyTorch implementation of BrainOOD from the paper 
*"	
BrainOOD: Out-of-distribution Generalizable Brain Network Analysis"* published in 13th International Conference on Learning Representations (ICLR'25).

## Current Branch Notes

This repository has been adapted for local ABIDE experiments. On `feature/remove-topo-cwn`, the extra topo, cell-complex, and CWN experiment path is removed again to stay closer to the original BrainOOD graph pipeline.

- `GOODABIDE` now returns standard PyG graph data only: `x`, `edge_index`, `edge_weight`, `y`, and `env_id`.
- All CWN and cell-complex-specific runtime inputs have been removed, including `x_0/x_1/x_2`, incidence matrices, adjacency matrices, and custom topo batching.
- The model path no longer initializes CWN-related channel settings or TopoModelX-based code.
- The remaining experiment flow keeps the current ABIDE dataset path and graph-level BrainOOD training stack.
- The ABIDE preprocessing path now reads node timeseries from the original `*_features_timeseries.mat` files, uses the original `*_correlation_matrix.mat` files for graph construction, and keeps the precomputed `feat` field as the runtime node feature input.
- The current branch now prefers the older sliding/time preprocessing artifact at `GOOD/data/bin_time_dataset/abide.bin` when it exists, so experiments can reuse the earlier memory-safe preprocessing path instead of the newer `bin_dataset` output.
- `GOOD/data/dataPreprocessing_sliding.py` is aligned with the earlier local preprocessing logic again: z-score timeseries features, `edge_ratio=0.2`, no default wavelet compression, and output written to `bin_time_dataset`.
- The pretraining stage now supports a separate effective batch size (`train.pretrain_bs`) and splits oversized train batches into smaller contrastive micro-batches before the two-view forward pass, so the CNN+LSTM encoder does not OOM before finetuning starts.
- The contrastive pretraining path on `feature/remove-topo-cwn` now uses cross-scale positive pairs: the original brain graph and its granular-ball coarse graph. The pretrain loss is applied to `GDGMT/GDGMTvGIN` projection-head outputs instead of classifier logits.
- The default pretrain config is now less aggressive for cross-scale contrast: `ood.temperature=0.07`, `train.pretrain_bs=32`, `train.pre_lr=5e-4`, `train.ball_r=0.3`, and `train.pre_epoch=50`, so the granular-ball view is coarser and pretraining stops near the observed contrastive-loss plateau instead of spending another 70 epochs polishing a nearly saturated objective.
- Granular-ball preprocessing logs are now muted during training; batched coarse-view construction reports progress through a single progress bar instead of per-graph debug prints.
- Metadata-V5 CSV metadata can now be converted directly into GOOD `meta.json` files for `neurocon`, `ppmi`, and `taowu` via `python3 GOOD/data/good_datasets/generate_metadata_v5_meta.py`; the generated metadata follows ABIDE-style keys (`idx2sex`, `idx2age`, `idx2site`, `idx2label`) with single-site `idx2site=0` and binary labels `Control=0`, non-control=1.

## Run

```bash
goodtg --config_path GOOD_configs/GOODABIDE/site/concept/BrainOOD.yaml
```

For this feature branch on the shared server, prefer running from the repo root so the current checkout is used instead of an older installed console script:

```bash
PYTHONPATH=$PWD python -m GOOD.kernel.main --config_path GOOD_configs/GOODABIDE/site/concept/BrainOOD.yaml
```

The default startup path now also auto-selects an almost-idle physical GPU when `CUDA_VISIBLE_DEVICES` is not already set. It maps the chosen physical card into the process and then runs on logical `cuda:0`, so even the original `goodtg` or `python -m GOOD.kernel.main ...` command follows the same GPU-selection rule. The launcher no longer uses a file lock during device selection, so concurrent invocations will not block on a shared lock file.

To avoid GPU OOM on a shared server, use the auto-GPU launcher instead of hard-coding `gpu_idx`. It picks an almost-idle physical GPU, exports `CUDA_VISIBLE_DEVICES` to that card, and forces the framework-side `gpu_idx` to `0` so device numbering stays consistent inside the process:

```bash
./scripts/run_brainood_auto_gpu.sh
```

You can override the selection threshold when the server is busier than usual:

```bash
MIN_FREE_MB=16000 MAX_USED_MB=2000 MAX_UTIL=20 ./scripts/run_brainood_auto_gpu.sh
```

## Contact

If you have any questions, please feel free to reach out at `jiaxing003@e.ntu.edu.sg`.


## Recent Changes (2026-04-14)

### Input Representation Overhaul
- PCA node features: Node features are now PCA(FC, 100->32) instead of raw FC rows, decoupling node features from edge information.
- Soft edge construction: Ledoit-Wolf shrinkage covariance + |w|>0.05 threshold replaces the old top-20% hard threshold, retaining ~74% of edges.
- Preprocessing: edge_ratio changed from 0.2 to 1.0 (complete graph) in both dataPreprocessing.py and dataPreprocessing_sliding.py.

### Granular-Ball Cross-Reweight (GBCR)
- Replaced VICReg contrastive branch with GBCR module embedded inside the GNN.
- GBCR sits between GAT layer 1 and layer 2: forms K=14 soft granular balls from mid-level representations, estimates ball-level node/edge importance, and maps importance back to the original graph as residual reweighting.
- Layer 2 changed from GATConv to GATv2Conv(edge_dim=1) to accept edge importance from GBCR.
- Node reweight: h+ = h * (1 + alpha_n * u), where u = Qr (ball importance mapped to nodes).
- Edge reweight: attention bias from ball-ball importance S mapped via M = QSQ^T.
- Domain generalization: class-conditional ball importance alignment across sites (L_gb-dg).
- Assignment entropy loss encourages balanced ball membership.

### Training Strategy (4 Strategic Decisions)
1. Stage-2-only main results: Stage 1+2 (classifier + site calibration) produce main results; stage 3 (GB/VICReg contrastive) is ablation-only.
2. MA5 checkpoint selection: Uses 5-epoch moving average of S_t = 0.6*BA_OOD_val + 0.2*AUROC_OOD_val + 0.2*BA_ID_val. BA (balanced accuracy) better handles low positive-ratio sites; AUROC is threshold-invariant.
3. Local-peak top-3 ensemble: Only saves checkpoints at local S_t peaks with min 4-epoch gap for diversity, then takes top-3 by MA5. Test-time averages softmax probs with threshold tuned on OOD_val to maximize balanced accuracy.
4. Stage-3 gating: GB/VICReg stage only enters main results if MA5 > best_stage2 + 0.02 for 3 consecutive epochs.


## Site Calibration (v2)

The sample-level statistical-aware calibration module has been refactored:

- **Position**: Moved from input (before GNN) to between GAT layer 1 and layer 2 (mid-level H₁)
- **Form**: Constrained affine — γ ∈ [0.9, 1.1], β ∈ [-0.1, 0.1], with fixed α=0.1 residual scaling
- **Input**: Meta-network now sees both node feature stats AND edge distribution stats (mean/std of |A|, density, positive edge ratio)
- **Training**: 3-stage schedule — gate-only (0-10), +SiteAdv (10-20), freeze gate + GBCR (20+)
- **Losses**: L_aff (affine constraint) + L_sp (gate sparsity) + L_gate-align (class-conditional gate alignment)
- **Monitoring**: Every epoch prints mean(g), std(g), ||γ-1||, ||β|| for gate interpretability


## Episodic Meta-Learning (site-level MLDG, branch `meta-learning`)

Adds a first-order, MLDG-style episodic training loop on top of the existing
`SiteCalibration + GBCR + SiteAdv` stack. Each training step now performs:

1. Pick one **source site** as the held-out *meta-test domain*; the rest are
   the *meta-train domains*.
2. Pull a batch from each meta-train site → forward → compute the full loss
   (cls + gate/affine/gate-align + SiteAdv + GBCR), accumulate, average.
3. Pull a batch from the meta-test site → forward → compute either cls-only
   (default) or the full loss, controlled by `meta_test_aux`.
4. `loss_total = mean(meta_train losses) + λ_meta * meta_test loss`.
5. Backward + grad-clip + optimizer step + post-step parameter freezes.

**Important — no data leakage.** `meta_test_site` is sampled only from the
sites in the *training* split (`self.loader['train'].dataset`). The real
OOD val / OOD test loaders are not part of `site_loaders`, so unseen
domains can never enter training.

### Code structure

- `GOOD/kernel/pipelines/episodic_meta.py` — helpers:
  - `build_site_loaders(train_dataset, batch_size, num_workers, seed)`
  - `sample_episode(source_sites, rng)`
  - `next_site_batch(site_iterators, site_loaders, site)` (auto-resets on
    iterator exhaustion → cyclic sampling for small sites).
- `GOOD/kernel/pipelines/basic_pipeline.py`:
  - `Pipeline._compute_total_loss(data, allow_aux)` — single source of
    truth for the training loss; reused by both `train_batch` and the
    episodic step so the loss formulation cannot drift.
  - `Pipeline._train_episode(...)` — one episodic step (first-order, no
    model cloning, no `create_graph=True`).
  - `Pipeline.train()` — auto-builds per-site DataLoaders when
    `use_episodic_meta` is on and dispatches to `_train_episode` for
    each step; otherwise the existing standard loop runs unchanged.
- `configs/GOOD_configs/GOODABIDE/site/concept/BrainOOD.yaml` — new keys:

  ```yaml
  ood:
    use_episodic_meta: false       # master switch
    lambda_meta: 0.5               # weight on meta-test loss
    meta_test_aux: false           # if true, meta-test also includes aux losses
    min_sites_per_episode: 2       # auto-fallback to standard if <2 source sites
  ```

### Episode example (4 source sites `[A, B, C, D]`)

```text
Episode 1: meta_train=[A,B,C]   meta_test=D
Episode 2: meta_train=[A,B,D]   meta_test=C
Episode 3: meta_train=[A,C,D]   meta_test=B
Episode 4: meta_train=[B,C,D]   meta_test=A
```

`meta_test_site` is uniformly sampled at random per step. Real OOD test
sites are never part of this rotation.

### Logging

Per epoch, additionally to the existing `Train / ID_val / OOD_val /
OOD_test / Gate / GBCR / SiteAdv` lines, each episode prints:

```text
[MetaEpisode] meta_test_site=<id> meta_train_sites=[...]
              loss_meta_train=... loss_meta_test=... loss_total=...
```

`spec_loss` for the epoch contains `MetaTrain` and `MetaTest` entries on
top of the existing `Align / Entropy / SiteAdv / GateAlign`.

### Notes

- First-order only. No `higher`, no `fast_weights`, no second-order
  gradients.
- Validation, test, evaluation, checkpoint NMS, MA5 ensemble, and
  threshold tuning are unchanged.
- If only one training source site exists, the pipeline prints a warning
  and falls back to standard training.
- Batch size for each site is `min(train_bs, site_size)`; small sites are
  cycled via `next_site_batch` so they keep contributing every episode.


## Performance / correctness fixes (meta-learning branch, 2026-05-07)

Three follow-up changes on top of the episodic meta-learning scaffolding.
All are local to existing modules — no new files, no API change for callers.

### 1. Single-round forward in `GDGMT` (was `sampling_rounds=3`)

`GOOD/networks/models/GDGMT.py` — the `while len(sampling_logits) < 3` loop
in `forward()` is removed. Each round ran the same forward with the same
inputs (only randomness was dropout / meta-net noise), so averaging gave
near-zero variance reduction while tripling the forward cost — particularly
painful in episodic mode, where one step already runs `(K-1) + 1` forwards.
The class still accepts `config.ood.extra_param[3]` for backward-compat but
the value is now ignored.

### 2. `site_adv` disabled in episodic mode

`GOOD/kernel/pipelines/basic_pipeline.py`:
- `_compute_total_loss(... enable_site_adv: bool = True)` gates the SiteAdv
  block on this flag.
- `_train_episode` always passes `enable_site_adv=False` for both
  meta-train and meta-test forwards.

Reason: each meta-train batch contains a single site, so cross-entropy on a
constant site label is degenerate (`H = log(num_sites)` plus tiny noise) and
gives no useful gradient. Standard non-episodic training is unaffected
(default flag `True`).

### 3. `SiteCalibration._compute_edge_stats` vectorised

`GOOD/networks/models/SiteCalibration.py` — the `for g in range(num_graphs)`
loop is replaced by `scatter_add_` aggregation, eliminating ~64 GPU-sync
points per batch. Mean / density / pos_ratio match the old loop exactly;
`std` switches from unbiased (`n-1`) to population (`sqrt(E[w²]-E[w]²)`),
which makes single-edge graphs return 0 instead of NaN. The relative
difference is `sqrt(n/(n-1))` which is < 0.02% for graphs with > 1000 edges
and is washed out by the downstream `LayerNorm` anyway.


## Correctness fixes (meta-learning branch, 2026-05-07 cont.)

### 4. Single source of truth for gate / affine regularization (`C2`)

`GOOD/ood_algorithms/algorithms/BrainOOD.py` — `loss_postprocess` no longer
adds `gate_sparsity_weight·gate_reg + affine_reg_weight·affine_reg`. Those
terms are kept only in `Pipeline._compute_total_loss`, which already applies
them with a 5-epoch ramp starting at `gate_reg_warmup`. Previously both
sites added the same terms, so once `ramp` saturated to 1 the effective
weight was 2x the configured value — i.e. all prior runs had been training
under `gate_sparsity_weight=0.10` and `affine_reg_weight=0.010` instead
of the YAML-declared `0.05 / 0.005`. Subsequent runs will use the
configured values exactly.

### 5. `mixed_val/test_score` written into training-time checkpoints (`C3`)

`GOOD/kernel/pipelines/basic_pipeline.py:_build_ckpt` — adds:

```python
'mixed_val_score':  (val_score·val_n + id_val_score·id_val_n) / (val_n + id_val_n),
'mixed_test_score': (test_score·test_n + id_test_score·id_test_n) / (test_n + id_test_n),
```

so the ckpts produced during `Pipeline.train()` carry the same fields as
`save_epoch`. Without them, `compute_10fold_metrics` fell back to
`val_score` (= `ood_val_score`) and the "Val / Test" columns in
`logs/grid_results.xlsx` silently reported OOD-only numbers, not the
ID+OOD mix the column header implied.
