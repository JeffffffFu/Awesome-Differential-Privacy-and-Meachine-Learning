from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt


def extract_node_subgraphs(graph: Data, num_hops, dataset_name: str = None):
    """
    Extract enclosing subgraphs for every node in the graph.

    Args:
    :param graph : Data contains attributes edge_index, x, y
    :param num_hops : number of hops
    """
    data_list = []
    # node_index = graph.edge_index.unique()
    # for src in node_index.t().tolist():  # The times of loop execution = #links = #samples
    # assert sorted(graph.edge_index.unique().tolist()) == list(range(graph.x.shape[0])), "node index does not match to node nums!" TODO
    for src in range(graph.x.shape[0]):
        src_origin = src
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src], num_hops, graph.edge_index,
            relabel_nodes=True)  # relabel Ture, label index in sub_edge_index will change.
        data = Data(x=graph.x[sub_nodes], src=src_origin, edge_index=sub_edge_index, y=graph.y[src],
                    sub_nodes=sub_nodes)  # sub nodes are original index.
        if dataset_name == "cora":
            data["train_mask"] = graph.train_mask[src]
            data["val_mask"] = graph.val_mask[src]
            data["test_mask"] = graph.test_mask[src]
        data_list.append(data)
    return data_list
