# BrainOOD

This repository is based on the official PyTorch implementation of BrainOOD from the paper
*"BrainOOD: Out-of-distribution Generalizable Brain Network Analysis"* published at ICLR 2025.

The original public release is:

- https://github.com/AngusMonroe/BrainOOD

## 当前改动总结（中文）

这个仓库已经不是原始 BrainOOD 代码的直接镜像，而是基于原论文代码继续演化出来的 ABIDE 脑网络研究分支。

目前的主要改动方向包括：

- 面向 ABIDE 数据的本地化数据加载、二进制数据读取和时序/滑窗预处理。
- 为脑图构建增加了额外的图预处理与转换逻辑，包括 granular-ball 相关脚本和 graph-to-cell-complex 工具。
- 引入了一批 topology / cell-complex 相关模块，用于支持 CWN 风格的实验。
- 修改了原始 BrainOOD 的模型与训练流程，包括 `GDGMT`、`GAT`、时序编码、分阶段预训练/微调、10-fold 运行、超参搜索和 t-SNE 可视化。

因此，这个仓库更适合被理解为“基于 BrainOOD 的研究开发版本”，而不是论文官方代码的严格复现版本。

## Current Project Summary

This codebase is no longer a plain mirror of the official BrainOOD release. It has been adapted into an ABIDE-focused research repository with several local experimental extensions.

Compared with the original BrainOOD code, the current repository mainly includes:

- ABIDE-oriented data loading and preprocessing code, including local binary dataset loading, temporal/sliding-window preprocessing scripts, and dataset conversion utilities.
- Additional graph construction and preprocessing helpers, including granular-ball related scripts and graph-to-cell-complex conversion utilities.
- Extensions around topological / cell-complex modeling, including local `GOOD/utils/cw`, `GOOD/utils/data`, and `GOOD/utils/mp` utility modules used to support CWN-style experiments.
- Model-side modifications to the original BrainOOD pipeline, including changes around `GDGMT`, `GAT`, and alternative feature extraction paths.
- Experimental temporal encoding support, including CNN/LSTM-style processing for node time-series features.
- Training-pipeline changes such as 10-fold execution, grid-search style hyperparameter sweeping, staged pretrain/finetune experiments, and t-SNE embedding visualization.
- Metric and evaluation adjustments for the current experimental setup.

## Important Difference From The Original Paper Code

The current repository should be treated as a research-development branch built on top of BrainOOD rather than a strict reproduction of the official paper release.

In particular:

- The data pipeline has been changed to fit local ABIDE experiments.
- The model stack is no longer restricted to the original BrainOOD configuration.
- Some OOD-loss and training behaviors have been modified or ablated during experimentation.
- Several modules are still experimental and are intended for iterative research rather than polished public release.

## Current Focus

At the moment, this repository is mainly used for:

- ABIDE site-level OOD experiments
- Brain graph representation learning
- Topology-aware / cell-complex based extensions
- Two-stage pretraining and finetuning exploration

## Run

Typical entrypoint:

```bash
goodtg --config_path GOOD_configs/GOODABIDE/site/concept/BrainOOD.yaml
```

## Contact

For questions about the original BrainOOD paper/codebase, please refer to the official repository and paper authors.
