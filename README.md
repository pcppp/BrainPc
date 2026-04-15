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
