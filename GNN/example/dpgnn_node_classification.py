import numpy as np
from torch_geometric.datasets import Planetoid, TUDataset
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from optimizer.dp_optimizer import DPSGD, DPAdam


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)  # 样本属性维度*embedding维度
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # 激活函数
        x = F.dropout(x, training=self.training)  # 只在training状态下进行
        x = self.conv2(x, edge_index)

        # readout
        # center pooling
        if isinstance(data, Batch):
            _, center_indices = np.unique(data.batch.cpu().numpy(), return_index=True)
            # sum pooling
            # x = global_add_pool(x, data.batch)
            x = x[center_indices]
        elif isinstance(data, Data):
            center_indices = 0,  # tuple, center pooling for one node subgraph
            x = x[center_indices].reshape(1, -1)  # ensure the shape is 2-dim

        return F.log_softmax(x, dim=1)  # 使用softmax概率归一


def train_val_test_split(subgraphs):
    train_subgraphs, val_subgraphs, test_subgraphs, other_graphs = [], [], [], []
    for graph in subgraphs:
        if graph.train_mask:
            train_subgraphs.append(graph)
        elif graph.val_mask:
            val_subgraphs.append(graph)
        elif graph.test_mask:
            test_subgraphs.append(graph)
        else:
            other_graphs.append(graph)
    print(f"{len(other_graphs)} graphs")
    return train_subgraphs, val_subgraphs, test_subgraphs


def main(batch_size, k_hop, max_norm, target_delta, max_terms_per_node, sigma, learning_rate, momentum):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # model = GCN(16, 2, dataset.num_node_features, dataset.num_classes).to(device)  # 模型复制到device上（就是参数需要复制）
    model = GCN()
    # data = dataset[0].to(device)
    data = dataset[0]
    from GNN.utils import extract_node_subgraphs
    import GNN.sampler as sampler
    data = sampler.subsample_graph(data, max_degree=5)
    subgraphs = extract_node_subgraphs(data, k_hop, "cora")
    train_subgraphs, val_subgraphs, test_subgraphs = train_val_test_split(subgraphs)
    # train_loader = DataLoader(data, batch_size=batch_size, num_workers=0, shuffle=True)
    # test_loader = DataLoader(data, batch_size=batch_size, num_workers=0, shuffle=True)
    # train_loader = DataLoader(train_subgraphs, batch_size=batch_size, num_workers=0, shuffle=True)
    # test_loader = DataLoader(test_subgraphs, batch_size=batch_size, num_workers=0, shuffle=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 使用Adam，weight_decay其实就是L2正则化
    optimizer = DPSGD(
        l2_norm_clip=max_norm,  # 裁剪范数
        noise_multiplier=sigma,
        minibatch_size=batch_size,  # 几个样本梯度进行一次梯度下降
        microbatch_size=1,  # 几个样本梯度进行一次裁剪，这里选择逐样本裁剪
        params=model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )
    model.train()  # 进入训练模式，和evaluate模式不同的是，evaluate模式会将很多参数的gradient置为不可求导
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = F.nll_loss
    for epoch in range(200):
        indices = np.random.choice(range(len(train_subgraphs)), size=(batch_size,), replace=False)
        train_batch_subgraphs = [train_subgraphs[i] for i in indices]
        train_loader = DataLoader(train_batch_subgraphs, batch_size=batch_size, num_workers=0, shuffle=False)
        from train_and_validation.train_with_dp import train_dynamic_add_noise_geo_uniform_batch
        train_loss, train_acc = train_dynamic_add_noise_geo_uniform_batch(model, train_loader, optimizer, criterion,full_batch=False)
        # train_loss, train_acc = train_dynamic_add_noise_geo_uniform_batch(model, train_loader, optimizer, criterion,
        #                                                                   full_batch=True)
        print(f"epoch:{epoch}, total loss:{train_loss}")

        # ------------------- privacy accounting ------------------- #
        from privacy_analysis.RDP.compute_multiterm_rdp import compute_multiterm_rdp
        from privacy_analysis.RDP.rdp_convert_dp import compute_eps
        orders = np.arange(1, 10, 0.1)[1:]
        rdp_every_epoch = compute_multiterm_rdp(orders, epoch, sigma, len(train_subgraphs),
                                                max_terms_per_node, batch_size)
        epsilon, best_alpha = compute_eps(orders, rdp_every_epoch, target_delta)
        print("epoch: {:3.0f}".format(epoch) + " | epsilon: {:7.4f}".format(
            epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))

    test_loader = DataLoader(test_subgraphs, batch_size=batch_size, num_workers=0, shuffle=True)
    from train_and_validation.validation import validation_geo_uniform_batch
    acc = validation_geo_uniform_batch(model, test_loader)
    print(f"Acc:{acc:.4f}")


if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    learning_rate = 0.5
    numEpoch = 15000
    sigma = 1.23
    momentum = 0.9
    delta = 10 ** (-5)
    max_norm = 0.1
    max_terms_per_node = 20
    target_delta = 1e-5

    # learning_rate = 0.5
    # numEpoch = 15000
    # sigma = 1e-5
    # momentum = 0.9
    # delta = 10 ** (-5)
    # # max_norm = 0.1
    # max_norm = 1000.0
    # max_terms_per_node = 20
    # target_delta = 1000.0
    main(32,
         1,
         max_norm=max_norm,
         sigma=sigma,
         target_delta=target_delta,
         max_terms_per_node=max_terms_per_node,
         learning_rate=learning_rate,
         momentum=momentum)
