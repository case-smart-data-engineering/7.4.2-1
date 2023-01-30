import scipy.sparse as sp
import numpy as np
import torch
import time
import os
from configparser import ConfigParser

import sys
sys.path.append(r'/workspace/7.4.2-1/1_算法示例/')
from util.load_data import load_data_with_features, load_data_without_features, sparse_to_tuple, mask_test_edges, preprocess_graph
from util.loss import gae_loss_function, vgae_loss_function
from util.metrics import get_roc_score
from util import define_optimizer
from gae.model import GCNModelVAE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train():
    def __init__(self):
        pass

    def train_model(self, config_path):
        if os.path.exists(config_path) and (os.path.split(config_path)[1].split('.')[0] == 'config') and (
                os.path.splitext(config_path)[1].split('.')[1] == 'cfg'):
            # 加载配置文件
            config = ConfigParser()
            config.read(config_path,encoding='utf-8')
            section = config.sections()[0]

            # 数据存储目录
            data_catalog = config.get(section, "data_catalog")

            # 节点路径
            node_cites_path = config.get(section, "node_cites_path")
            node_cites_path = os.path.join(data_catalog, node_cites_path)

            # 节点特征路径
            node_features_path = config.get(section, 'node_features_path')
            node_features_path = os.path.join(data_catalog, node_features_path)

            # 训练结果存储位置和加载模型结果
            model_path = config.get(section, "model_path")

            # 模型默认参数
            with_feats = config.getboolean(section, 'with_feats') 
            hidden_dim1 = config.getint(section, "hidden_dim1")
            hidden_dim2 = config.getint(section, "hidden_dim2")
            dropout = config.getfloat(section, "dropout")
            vae_bool = config.getboolean(section, 'vae_bool')
            lr = config.getfloat(section, "lr")
            lr_decay = config.getfloat(section, 'lr_decay')
            weight_decay = config.getfloat(section, "weight_decay")
            gamma = config.getfloat(section, "gamma")
            momentum = config.getfloat(section, "momentum")
            eps = config.getfloat(section, "eps")
            clip = config.getfloat(section, "clip")
            epochs = config.getint(section, "epochs")
            optimizer_name = config.get(section, "optimizer")

            if with_feats:
                # 加载带节点特征的数据集
                adj, features = load_data_with_features(node_cites_path, node_features_path)
            else:
                # 加载不带节点特征的数据集
                adj = load_data_without_features(node_cites_path)
                features = sp.identity(adj.shape[0])
            num_nodes = adj.shape[0]
            num_edges = adj.sum()

            features = sparse_to_tuple(features)
            num_features = features[2][1]

            # 去除对角线元素
            adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj_orig.eliminate_zeros()

            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_orig)

            adj = adj_train

            adj_norm = preprocess_graph(adj)
            adj_label = adj_train + sp.eye(adj_train.shape[0])
            adj_label = torch.FloatTensor(adj_label.toarray()).to(DEVICE)
            pos_weight = float(adj.shape[0] * adj.shape[0] - num_edges) / num_edges
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

            # 创建模型
            print('create model ...')
            model = GCNModelVAE(num_features, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout, vae_bool=vae_bool)

            # 定义优化器
            if optimizer_name == 'adam':
                optimizer = define_optimizer.define_optimizer_adam(model, lr=lr, weight_decay=weight_decay)

            elif optimizer_name == 'adamw':
                optimizer = define_optimizer.define_optimizer_adamw(model, lr=lr, weight_decay=weight_decay)

            elif optimizer_name == 'sgd':
                optimizer = define_optimizer.define_optimizer_sgd(model, lr=lr, momentum=momentum,
                                                                  weight_decay=weight_decay)

            elif optimizer_name == 'adagrad':
                optimizer = define_optimizer.define_optimizer_adagrad(model, lr=lr, lr_decay=lr_decay,
                                                                      weight_decay=weight_decay)

            elif optimizer_name == 'rmsprop':
                optimizer = define_optimizer.define_optimizer_rmsprop(model, lr=lr, weight_decay=weight_decay,
                                                                      momentum=momentum)

            elif optimizer_name == 'adadelta':
                optimizer = define_optimizer.define_optimizer_adadelta(model, lr=lr, weight_decay=weight_decay)

            else:
                raise NameError('No define optimization function name!')

            model = model.to(DEVICE)
            # 稀疏张量被表示为一对致密张量：一维张量和二维张量的索引。可以通过提供这两个张量来构造稀疏张量
            adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                torch.FloatTensor(adj_norm[1]),
                                                torch.Size(adj_norm[2]))
            features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                                torch.FloatTensor(features[1]),
                                                torch.Size(features[2])).to_dense()
            adj_norm = adj_norm.to(DEVICE)
            features = features.to(DEVICE)
            norm = torch.FloatTensor(np.array(norm)).to(DEVICE)
            pos_weight = torch.tensor(pos_weight).to(DEVICE)
            num_nodes = torch.tensor(num_nodes).to(DEVICE)

            print('start training...')
            best_valid_roc_score = float('-inf')
            hidden_emb = None
            model.train()
            for epoch in range(epochs):
                t = time.time()
                optimizer.zero_grad()
                recovered, mu, logvar = model(features, adj_norm)
                if vae_bool:
                    loss = vgae_loss_function(preds=recovered, labels=adj_label,
                                              mu=mu, logvar=logvar, n_nodes=num_nodes,
                                              norm=norm, pos_weight=pos_weight)
                else:
                    loss = gae_loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                cur_loss = loss.item()
                optimizer.step()

                hidden_emb = mu.data.cpu().numpy()
                # 评估验证集，val set
                roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
                # 保存最好的roc score
                if roc_score > best_valid_roc_score:
                    best_valid_roc_score = roc_score
                    # 不需要保存整个model，只需保存hidden_emb，因为后面的解码是用hidden_emb内积的形式作推断
                    np.save(model_path, hidden_emb)

                print("Epoch:", '%04d' % (epoch + 1), "train_loss = ", "{:.5f}".format(cur_loss),
                      "val_roc_score = ", "{:.5f}".format(roc_score),
                      "average_precision_score = ", "{:.5f}".format(ap_score),
                      "time=", "{:.5f}".format(time.time() - t)
                      )

            print("Optimization Finished!")

            # 评估测试集，test set
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            print('test roc score: {}'.format(roc_score))
            print('test ap score: {}'.format(ap_score))

        else:
            raise FileNotFoundError('File config.cfg not found : ' + config_path)

if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    train = Train()
    train.train_model(config_path)
