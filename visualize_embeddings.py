"""
可视化验证脚本：使用t-SNE对比未训练和训练后的模型embedding
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
import argparse

print("Visualization script created successfully!")
