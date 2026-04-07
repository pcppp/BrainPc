# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BrainOOD** is a PyTorch/PyG framework for out-of-distribution (OOD) generalizable brain network analysis

It extends a novel BrainOOD algorithm using graph diffusion and mixture-of-Gaussian pooling.

## Research Context

This repo should be treated as an evolving BrainOOD-derived academic codebase rather than generic BrainOOD maintenance.

- Use `brainOOD-origin` as the conceptual baseline when comparing behavior or explaining modifications.
- The active direction is to replace BrainOOD's original site-gap reduction method with a new model design for cross-site domain generalization.
- The scientific goal is to reduce site/scanner bias while preserving disease-relevant signals.
- Evaluate changes by the final 10-fold cross-validation metrics, not by single-fold logs, training loss, or short terminal excerpts.

## Working Rules

- Inspect the current branch and repo state before editing.Unless the user explicitly redirects the work, assume the main research line continues on `remove-topo-cwn`.
- Ignore redundant code in the project (dead code unrelated to the main process).Focus on the code related to the main process.
- Prefer focused model-path edits over peripheral refactors.
- For every code update, also update `README.md` so the current change scope is documented.
- Prefer narrow validation unless the user explicitly asks for a full experiment run.
- When reporting experiment results, prefer the newest saved 10-fold summary in `logs/grid_results.xlsx`.

## Remote SSH Context

**所有代码修改必须在远端服务器上进行，禁止直接编辑本地文件。** 本地仓库仅作为参考，远端服务器是唯一权威工作区。

- SSH host: `guest@172.24.2.25`
- SSH port: `5122`
- 远端工作目录: `/home/guest/workplace/pc/BrainOOD3(Moe)`
- Python 环境: `pc2`

**每次任务开始前，必须执行以下流程：**

```bash
ssh -p 5122 guest@172.24.2.25
cd '/home/guest/workplace/pc/BrainOOD3(Moe)'
conda activate pc2
```

- 使用 Bash 工具通过 SSH 在远端执行所有读取、编辑、验证操作。
- 读文件用 `ssh -p 5122 guest@172.24.2.25 "cat /home/guest/workplace/pc/BrainOOD3\(Moe\)/path/to/file"`。
- 编辑文件用 `ssh -p 5122 guest@172.24.2.25 "cd '/home/guest/workplace/pc/BrainOOD3(Moe)' && <edit command>"`，或通过 heredoc 写入。
- 运行验证命令时加 `conda run -n pc2` 前缀，避免依赖 shell 激活状态。
- **严禁**使用本地 Read / Edit / Write 工具直接修改 `/Users/pennychang/Desktop/...` 下的源码文件（CLAUDE.md 本身除外）。

  

  
