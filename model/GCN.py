from torch_geometric.datasets import Planetoid, TUDataset
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
# from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        # self.conv1 = GCNConv(dataset.num_node_features, 16)  # 样本属性维度*embedding维度
        # self.conv2 = GCNConv(16, dataset.num_classes)
        self.conv1 = GCNConv(num_node_features, 16)  # 样本属性维度*embedding维度
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # 激活函数
        x = F.dropout(x, training=self.training)  # 只在training状态下进行
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # 使用softmax概率归一


class GCNGraphClassification(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCNGraphClassification, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
