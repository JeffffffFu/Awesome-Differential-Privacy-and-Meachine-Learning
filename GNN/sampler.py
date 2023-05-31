from torch_geometric.data import Data
import torch
import numpy as np


def reverse_edges(edges):
    """Reverses an edgelist to obtain incoming edges for each node."""
    reversed_edges = {u: [] for u in edges}
    for u, u_neighbors in edges.items():
        for v in u_neighbors:
            reversed_edges[v].append(u)
    return reversed_edges


def get_adjacency_lists(dataset: Data):
    """Returns a dictionary of adjacency lists, with nodes as keys."""

    edges = {u: [] for u in range(dataset.num_nodes)}
    for u, v in dataset.edge_index.T.tolist():
        edges[u].append(v)
    return edges


def get_adjacency_lists_ogb(graph: torch.Tensor):
    ids = graph.unique().tolist()
    edges = {u: [] for u in ids}
    for u, v in graph.T.tolist():
        edges[u].append(v)
    return edges


def sample_adjacency_lists(edges, train_nodes,
                           max_degree):
    """Statelessly samples the adjacency lists with in-degree constraints.

    This implementation performs Bernoulli sampling over edges.

    Note that the degree constraint only applies to training subgraphs.
    The validation and test subgraphs are sampled completely.

    Args:
      edges: The adjacency lists to sample.
      train_nodes: A sequence of train nodes.
      max_degree: The bound on in-degree for any node over training subgraphs.
      rng: The PRNGKey for reproducibility
    Returns:
      A sampled adjacency list, indexed by nodes.
    """
    train_nodes = set(train_nodes)
    all_nodes = edges.keys()

    reversed_edges = reverse_edges(edges)
    sampled_reversed_edges = {u: [] for u in all_nodes}

    # For every node, bound the number of incoming edges from training nodes.
    dropped_count = 0
    for u in all_nodes:
        # u_rng = jax.random.fold_in(rng, u)
        incoming_edges = reversed_edges[u]
        incoming_train_edges = [v for v in incoming_edges if v in train_nodes]
        if not incoming_train_edges:
            continue

        in_degree = len(incoming_train_edges)
        sampling_prob = max_degree / (2 * in_degree)
        # sampling_mask = (
        #         jax.random.uniform(u_rng, shape=(in_degree,)) <= sampling_prob)
        sampling_mask = (
                np.random.uniform(size=(in_degree,)) <= sampling_prob)
        sampling_mask = np.asarray(sampling_mask)

        incoming_train_edges = np.asarray(incoming_train_edges)[sampling_mask]
        unique_incoming_train_edges = np.unique(incoming_train_edges)

        # Check that in-degree is bounded, otherwise drop this node.
        if len(unique_incoming_train_edges) <= max_degree:
            sampled_reversed_edges[u] = unique_incoming_train_edges.tolist()
        else:
            dropped_count += 1

    print('dropped count', dropped_count)
    sampled_edges = reverse_edges(sampled_reversed_edges)

    # For non-train nodes, we can sample the entire edgelist.
    for u in all_nodes:
        if u not in train_nodes:
            sampled_edges[u] = edges[u]
    return sampled_edges


def get_train_indices(graph: Data):
    indices = torch.where(graph.train_mask)[0].tolist()
    return indices


def subsample_graph(graph: Data, max_degree):
    edges = get_adjacency_lists(graph)
    train_indices = get_train_indices(graph)
    edges = sample_adjacency_lists(edges, train_indices, max_degree)

    senders = []
    receivers = []
    for u in edges:
        for v in edges[u]:
            senders.append(u)
            receivers.append(v)
    edge_index = torch.tensor([senders, receivers])
    graph.edge_index = edge_index
    return graph


def subsample_graph_ogb(graph: torch.Tensor, max_degree):
    edges = get_adjacency_lists_ogb(graph)
    tmp = [len(l) for l in edges.values()]
    max_degree = np.max([len(l) for l in edges.values()])
    train_indices = graph.unique().tolist()
    edges = sample_adjacency_lists(edges, train_indices, max_degree)
    senders = []
    receivers = []
    for u in edges:
        for v in edges[u]:
            senders.append(u)
            receivers.append(v)
    edge_index = torch.tensor([senders, receivers])
    graph = edge_index
    return graph


def subsample_graph_for_undirected_graph(graph, max_degree):
    graph = subsample_graph_ogb(graph, max_degree)
    graph = graph[[1, 0]]
    graph = subsample_graph_ogb(graph, max_degree)
    return graph
