import numpy as np
from torch_geometric.datasets import Planetoid, TUDataset
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


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
        _, center_indices = np.unique(data.batch.cpu().numpy(), return_index=True)
        x = x[center_indices]
        # sum pooling
        # x = global_add_pool(x, data.batch)

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


def main(batch_size, k_hop):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # model = GCN(16, 2, dataset.num_node_features, dataset.num_classes).to(device)  # 模型复制到device上（就是参数需要复制）
    model = GCN()
    # data = dataset[0].to(device)
    data = dataset
    from GNN.utils import extract_node_subgraphs
    subgraphs = extract_node_subgraphs(data[0], k_hop, "cora")
    train_subgraphs, val_subgraphs, test_subgraphs = train_val_test_split(subgraphs)
    # train_loader = DataLoader(data, batch_size=batch_size, num_workers=0, shuffle=True)
    # test_loader = DataLoader(data, batch_size=batch_size, num_workers=0, shuffle=True)
    # train_loader = DataLoader(train_subgraphs, batch_size=batch_size, num_workers=0, shuffle=True)
    # test_loader = DataLoader(test_subgraphs, batch_size=batch_size, num_workers=0, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 使用Adam，weight_decay其实就是L2正则化
    model.train()  # 进入训练模式，和evaluate模式不同的是，evaluate模式会将很多参数的gradient置为不可求导
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        total_loss = []
        graphs_per_batch = []
        step = 0
        indices = np.random.choice(range(len(train_subgraphs)), size=(batch_size,), replace=False)
        train_batch_subgraphs = [train_subgraphs[i] for i in indices]
        # train_batch_subgraphs = np.array(train_subgraphs, dtype=Data)[indices]
        # train_batch_subgraphs = train_batch_subgraphs.tolist()
        train_loader = DataLoader(train_batch_subgraphs, batch_size=batch_size, num_workers=0, shuffle=True)
        for index, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()  # optimizer的梯度需要置0的原因
            # out = model(batch.x, batch.edge_index, batch.batch)  # 对所有数据做forward
            out = model(batch)
            # loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            total_loss.append(loss.item())
            graphs_per_batch.append(batch.num_graphs)
            optimizer.step()
            step += 1
            if step >= 2:
                raise ValueError("only one batch is allowed in train_loader") # TODO
        total_loss = sum(total_loss) / sum(graphs_per_batch)
        print(f"epoch:{epoch}, total loss:{total_loss}")

    model.eval()
    correct = 0
    test_loader = DataLoader(test_subgraphs, batch_size=batch_size, num_workers=0, shuffle=True)
    for index, batch in enumerate(test_loader):
        batch = batch.to(device)
        optimizer.zero_grad()  # optimizer的梯度需要置0的原因
        # out = model(batch.x, batch.edge_index, batch.batch)  # 对所有数据做forward
        out = model(batch)  # 对所有数据做forward
        pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())  # 正确分类的数量

    # acc = int(correct) / int(data.test_mask.sum())  # 正确分类数/总个数
    acc = int(correct) / len(test_subgraphs)  # 正确分类数/总个数
    print(f"Acc:{acc:.4f}")


if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    main(32, 1)
