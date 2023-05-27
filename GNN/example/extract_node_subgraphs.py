from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(edge_index, title: str = None):
    G = nx.Graph()
    G.add_edges_from([(r1, r2) for r1, r2 in zip(edge_index.numpy()[0], edge_index.numpy()[1])])
    nx.draw(G, cmap=plt.get_cmap('jet'), with_labels=True)
    plt.title(title)
    plt.show()


def print_data(data: Data, tag="DATA LOADED"):
    print("*" * 80 + f"{tag} OVERVIEW" + "*" * 80)
    print(*[f"{key}:{val.size()}" for key, val in data.items()], sep="\n")
    print("*" * 80 + f"{tag}" + "*" * 80)
    print(*[f"{key}\n{val}" for key, val in data.items()], sep="\n")


def load_data():
    edge_index = torch.tensor([[0, 2, 3, 5, 5, 7, 8, 10, 11, 12, 13, 14, 14, 14],
                               [1, 1, 2, 4, 6, 8, 9, 9, 10, 10, 12, 11, 9, 8]])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)

    n_feats = 10
    x = torch.randint(low=0, high=4, size=(edge_index.unique().size(0), n_feats))

    new_data = Data(edge_index=edge_index, x=x)
    print(f"edge_index.size:{edge_index.size()}")
    print(f"edge_index:\n{edge_index.T.data}")
    draw_graph(edge_index, "original graph")
    return new_data


def extract_node_subgraphs(df, node_index, edge_index, num_hops, y):
    """
    Extract enclosing subgraphs for every node in the graph.

    Args:
    :param node_index : nodes in the graph ranxun: TODO link_index should be the undoubellded type
    :param edge_index : edges in the graph
    :param num_hops : number of hops
    :param y : if true link
    """
    data_list = []
    for src in node_index.t().tolist():  # The times of loop execution = #links = #samples
        src_origin = src
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src], num_hops, edge_index,
            relabel_nodes=False)  # relabel Ture, label index in sub_edge_index will be changed.
        # sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        #     [src], num_hops, edge_index,
        #     relabel_nodes=True)  # relabel Ture, label index in sub_edge_index will be changed.
        # src = mapping.tolist()  # the new ids where src, dst are mapped into?

        # Remove target link from the subgraph.
        # mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)  # 对于每一条边，节点两端的(src,dst)对上了，就为False
        # mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)  # 和(dst,src)对上了，就为False
        # sub_edge_index = sub_edge_index[:, mask1 & mask2]  # 两种False情况满足一个，就不选中这个边

        data = Data(x=df.x[sub_nodes], src=src_origin, edge_index=sub_edge_index, y=y,
                    sub_nodes=sub_nodes)  # sub nodes are original index.
        data_list.append(data)

    return data_list


if __name__ == "__main__":
    new_data = load_data()
    subgraphs = extract_node_subgraphs(new_data, new_data.edge_index.unique(), new_data.edge_index, 2, 1)
