from torch_geometric.datasets import Planetoid, TUDataset, KarateClub
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Batch
from model.GCN import GCNGraphClassification as GCN


def load_dataset():
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    return dataset, train_dataset, test_dataset


def train(model, train_loader, criterion, optimizer):
    model.train()
    for data in train_loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def main(hidden_channels, batch_size, num_epochs, learning_rate=0.01, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(hidden_channels, dataset.num_node_features, dataset.num_classes).to(device)  # 模型复制到device上（就是参数需要复制）

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, train_dataset, test_dataset = load_dataset()
    main(hidden_channels=64, batch_size=64, num_epochs=200, learning_rate=0.01, weight_decay=5e-4)
