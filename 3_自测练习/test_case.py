#!/usr/bin/env python3
import os.path as osp
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from my_solution import Test,Net


# 测试用例
def test_solution():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'cora'  # 获取数据集并进行处理
    path = osp.join(osp.dirname(osp.realpath(__file__)))
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    dataset[0].edge_index.to(device)
    data = data.to(device)
    model = Net(dataset.num_features, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    z = Test(data, model) 
    print(z)
if __name__ == "__main__":
    test_solution()
