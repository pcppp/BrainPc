#!/bin/bash

# 查找 Conda 环境中的 libgomp.so.1 路径
LIBGOMP_PATH=$(find ~/miniconda3/envs/barinPc -name "libgomp.so.1" | head -n 1) goodtg --config_path GOOD_configs/GOODABIDE/site/concept/BrainOOD.yaml

# 使用 LD_PRELOAD 启动 Python 脚本或命令
LD_PRELOAD="$LIBGOMP_PATH" python "$@"
