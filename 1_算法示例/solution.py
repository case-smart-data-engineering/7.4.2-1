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



class Net(torch.nn.Module):#构建神经网络模型
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)# 头尾节点属性对应相乘后求和 返回一个 [(正样本数+负样本数),1] 的向量
    def decode_all(self, z):
        prob_adj = z @ z.t()# 头节点属性和尾节点属性对应相乘后求和，[节点数，节点数]
        return (prob_adj > 0).nonzero(as_tuple=False).t()

# 生成正负样本边的标记
def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

#训练
def train(data, model, optimizer):
    model.train()
    #正负采样
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    optimizer.zero_grad()
    # 节点representation learning
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)
    #损失计算
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def Test(data, model):#定义单个epoch训练和测试过程
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']#正edge_index
        neg_edge_index = data[f'{prefix}_neg_edge_index']#负edge_index
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)#有无边的概率预测
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)#真实有边的结点标签
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        results.append(average_precision_score(link_labels.cpu(), link_probs.cpu()))
    return results


def main():#训练、测试、验证的过程
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
    df = pd.DataFrame(columns=["Val", "Test", "Best_val"])
    df.index.name = "Epoch"
    best_val_auc = test_auc = 0
    for epoch in range(1, 101):
        loss = train(data, model, optimizer)
        val_auc, tmp_test_auc = Test(data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc
        df.loc[epoch, "Val"] = val_auc
        df.loc[epoch, "Test"] = test_auc
        df.loc[epoch, "Best_val"] = best_val_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')
    z = model.encode(data.x, data.train_pos_edge_index)
    print(model.decode_all(z))



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'cora'  # 获取数据集并进行处理
    path = osp.join(osp.dirname(osp.realpath(__file__)))
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    dataset[0].edge_index.to(device)
    data = data.to(device)
    model = Net(dataset[0].num_features,64).to(device)
    test_acc = Test(data,model)
    print('Test Accuracy:{:.4f},Val Accuracy:{:.4f}; Test Precision:{:.4f},Val Precision:{:4f} '.format(*test_acc))
