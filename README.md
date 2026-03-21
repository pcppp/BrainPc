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
- The ABIDE preprocessing path now keeps raw node timeseries in `N_features`, reads edge connectivity from the original `*_correlation_matrix.mat`, and uses the precomputed `feat` field as the runtime node feature input.

## Run

```bash
goodtg --config_path GOOD_configs/GOODABIDE/site/concept/BrainOOD.yaml
```

## Contact

If you have any questions, please feel free to reach out at `jiaxing003@e.ntu.edu.sg`.
