from torch_geometric.datasets import Planetoid, TUDataset, KarateClub
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Batch


# class GCNDataset(InMemoryDataset):
#     def __init__(self, data):
#         super(GCNDataset,self).__init__()
#         self.data = data

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
        return F.log_softmax(x, dim=1)  # 使用softmax概率归一


def main(batch_size, num_epochs, learning_rate=0.01, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)  # 模型复制到device上（就是参数需要复制）
    # data = dataset[0].to(device)
    # gcn_dataset = GCNDataset(data)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    # train_loader = NeighborLoader(dataset[0], num_neighbors=[30, 30], batch_size=batch_size,
    #                               input_nodes=dataset[0].train_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # 使用Adam，weight_decay其实就是L2正则化
    model.train()  # 进入训练模式，和evaluate模式不同的是，evaluate模式会将很多参数的gradient置为不可求导
    for epoch in range(num_epochs):
        for index, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()  # optimizer的梯度需要置0的原因
            out = model(data)  # 对所有数据做forward
            # out = out[data.ptr[:-1]]
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) # The negative log likelihood loss（基于train数据的loss）
            # loss = F.nll_loss(out[data.batch == index], data.y)
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) # input tensor
            # loss = F.nll_loss(out[data.ptr[:-1]], data.y)
            # tmp = data.ptr
            # loss = F.nll_loss(out[data.ptr[:-1]], data.y)
            # loss = F.nll_loss(out[[0]], data.y)  # batch_size=1
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch}, loss:{loss}")

    model.eval()
    correct = 0
    for index, data in enumerate(train_loader):
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        # correct += (pred[data.ptr[:-1]] == data.y).sum()
        correct += (pred[[0]] == data.y).sum()
    acc = int(correct) / len(dataset.data.y)
    # pred = model(data).argmax(dim=1)  # 取概率最高的结果的index作为pred的class
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()  # 正确分类的数量
    # # correct = (pred == data.y).sum()
    # # acc = int(correct) / int(data.test_mask.sum())  # 正确分类数/总个数
    # acc = int(correct) / int(data.test_mask.sum())  # 正确分类数/总个数
    print(f"Acc:{acc:.4f}")


if __name__ == "__main__":
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    # dataset = KarateClub()
    main(batch_size=32, num_epochs=200, learning_rate=0.01, weight_decay=5e-4)
