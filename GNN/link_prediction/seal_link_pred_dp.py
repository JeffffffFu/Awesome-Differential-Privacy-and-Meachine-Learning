# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
import os, sys
import os.path as osp
from shutil import copy
import copy as cp
from tqdm import tqdm
import pdb

import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import warnings

from privacy_analysis.RDP import compute_rdp

warnings.filterwarnings("ignore", category=UserWarning)

from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)

from utils import *
from models import *


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}_dp'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        from GNN.sampler import subsample_graph_for_undirected_graph
        from torch_geometric.utils import is_undirected, to_undirected
        if not is_undirected(pos_edge):
            pos_edge = to_undirected(pos_edge)
        if not is_undirected(neg_edge):
            neg_edge = to_undirected(neg_edge)
        pos_edge = subsample_graph_for_undirected_graph(pos_edge, max_degree=args.max_node_degree)
        neg_edge = subsample_graph_for_undirected_graph(neg_edge, max_degree=args.max_node_degree)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class SEALDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]
        tmp = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=self.data.x,
                             y=y, directed=self.directed, A_csc=self.A_csc)
        data = construct_pyg_graph(*tmp, self.node_label)

        return data


# def train():
#     model.train()
#
#     total_loss = 0
#     pbar = tqdm(train_loader, ncols=70)
#     for data in pbar:
#         data = data.to(device)
#         optimizer.zero_grad()
#         x = data.x if args.use_feature else None
#         edge_weight = data.edge_weight if args.use_edge_weight else None
#         node_id = data.node_id if emb else None
#         logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
#         loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs
#
#     return total_loss / len(train_dataset)


def train_dynamic_add_noise(model, train_loader, optimizer, criterion, full_batch=False):
    '''
    Args:
        model:
        train_loader: PyG DataLoader
        optimizer:

    Returns:

    '''
    model.train()
    train_loss = 0.0
    aa = 0
    train_acc = 0.
    i = 0

    for id, data in enumerate(train_loader):  # TODO per-sample computation for mini-batch
        optimizer.zero_accum_grad()  # 梯度清空
        for id in range(data.num_graphs):
            data_microbatch = data[id]
            data_microbatch.to(device)
            optimizer.zero_microbatch_grad()
            x = data_microbatch.x if args.use_feature else None
            edge_weight = data_microbatch.edge_weight if args.use_edge_weight else None
            node_id = data_microbatch.node_id if emb else None
            # out = model(data_microbatch.x, data_microbatch.edge_index, data_microbatch.batch)
            logits = model(data_microbatch.z, data_microbatch.edge_index, data_microbatch.batch, x, edge_weight,
                           node_id, micro_batch=True)
            loss = criterion(logits.view(-1), data_microbatch.y.to(torch.float))
            loss.backward()  # 梯度求导，这边求出梯度
            optimizer.microbatch_step()  # 这个step做的是每个样本的梯度裁剪和梯度累加的操作
            train_loss += loss.item()
        optimizer.step_dp()  # 这个做的是梯度加噪和梯度平均更新下降的操作
    return train_loss, train_acc  # 返回平均损失和平均准确率


def compute_max_terms_per_node(num_message_passing_steps, max_node_degree):
    max_node_degree = 2 * max_node_degree ** 2
    if num_message_passing_steps == 1:
        return max_node_degree

    if num_message_passing_steps == 2:
        return max_node_degree ** 2 + max_node_degree

    if num_message_passing_steps == 3:
        return max_node_degree ** 3 + max_node_degree * 2 + max_node_degree


def compute_base_sensitivity(num_message_passing_steps, max_degree):
    """Returns the base sensitivity which is multiplied to the clipping threshold.

    Args:

    """

    num_message_passing_steps = num_message_passing_steps
    max_node_degree = max_degree

    if num_message_passing_steps == 1:
        return float(2 * (max_node_degree + 1))

    if num_message_passing_steps == 2:
        return float(2 * (max_node_degree ** 2 + max_node_degree + 1))

    if num_message_passing_steps == 3:
        return float(2 * (max_node_degree ** 3 + max_node_degree * 2 + max_node_degree))

    # We only support MLP and upto 2-layer GNNs.
    raise ValueError('Not supported for num_message_passing_steps > 2.')


def train():
    model.train()
    criterion = BCEWithLogitsLoss()
    indices = np.random.choice(range(len(train_dataset)), size=(args.batch_size,), replace=False)
    train_batch_subgraphs = [train_dataset[i] for i in indices]
    train_loader = DataLoader(train_batch_subgraphs, batch_size=args.batch_size, num_workers=0, shuffle=False)
    train_loss, train_acc = train_dynamic_add_noise(model, train_loader, optimizer, criterion,
                                                    full_batch=False)
    print(f"epoch:{epoch}, total loss:{train_loss}")

    # ------------------- privacy accounting ------------------- #
    from privacy_analysis.RDP.compute_multiterm_rdp import compute_multiterm_rdp
    from privacy_analysis.RDP.rdp_convert_dp import compute_eps
    orders = np.arange(1, 10, 0.1)[1:]
    max_terms_per_node = compute_max_terms_per_node(num_message_passing_steps=args.num_layers,
                                                    max_node_degree=args.max_node_degree)
    rdp_every_epoch = compute_multiterm_rdp(orders, epoch, sigma, len(train_dataset),
                                            max_terms_per_node, args.batch_size)

    from privacy_analysis.RDP.compute_rdp import compute_rdp
    rdp_every_epoch_org = compute_rdp(args.batch_size / len(train_dataset), sigma, 1 * epoch, orders)
    epsilon, best_alpha = compute_eps(orders, rdp_every_epoch, target_delta)
    epsilon_org, best_alpha_org = compute_eps(orders, rdp_every_epoch_org, target_delta)
    print("epoch: {:3.0f}".format(epoch) + " | epsilon: {:10.7f}".format(
        epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))
    print("epoch: {:3.0f}".format(epoch) + " | epsilon_org: {:10.7f}".format(
        epsilon_org) + " | best_alpha: {:7.4f}".format(best_alpha_org))
    return train_loss


@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true == 1]
    neg_val_pred = val_pred[val_true == 0]

    y_pred, y_true = [], []
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true == 1]
    neg_test_pred = test_pred[test_true == 0]

    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'rocauc':
        results = evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results


@torch.no_grad()
def test_multiple_models(models):
    for m in models:
        m.eval()

    y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
    for data in tqdm(val_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, m in enumerate(models):
            logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred[i].append(logits.view(-1).cpu())
            y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    val_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    val_true = [torch.cat(y_true[i]) for i in range(len(models))]
    pos_val_pred = [val_pred[i][val_true[i] == 1] for i in range(len(models))]
    neg_val_pred = [val_pred[i][val_true[i] == 0] for i in range(len(models))]

    y_pred, y_true = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
    for data in tqdm(test_loader, ncols=70):
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        for i, m in enumerate(models):
            logits = m(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
            y_pred[i].append(logits.view(-1).cpu())
            y_true[i].append(data.y.view(-1).cpu().to(torch.float))
    test_pred = [torch.cat(y_pred[i]) for i in range(len(models))]
    test_true = [torch.cat(y_true[i]) for i in range(len(models))]
    pos_test_pred = [test_pred[i][test_true[i] == 1] for i in range(len(models))]
    neg_test_pred = [test_pred[i][test_true[i] == 0] for i in range(len(models))]

    Results = []
    for i in range(len(models)):
        if args.eval_metric == 'hits':
            Results.append(evaluate_hits(pos_val_pred[i], neg_val_pred[i],
                                         pos_test_pred[i], neg_test_pred[i]))
        elif args.eval_metric == 'mrr':
            Results.append(evaluate_mrr(pos_val_pred[i], neg_val_pred[i],
                                        pos_test_pred[i], neg_test_pred[i]))
        elif args.eval_metric == 'rocauc':
            Results.append(evaluate_ogb_rocauc(pos_val_pred[i], neg_val_pred[i],
                                               pos_test_pred[i], neg_test_pred[i]))

        elif args.eval_metric == 'auc':
            Results.append(evaluate_auc(val_pred[i], val_true[i],
                                        test_pred[i], test_pred[i]))
    return Results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)
    return results


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    results['AUC'] = (valid_auc, test_auc)

    return results


def evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })[f'rocauc']

    test_rocauc = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'rocauc']

    results = {}
    results['rocauc'] = (valid_rocauc, test_rocauc)
    return results


# Data settings
parser = argparse.ArgumentParser(description='OGBL (SEAL)')
parser.add_argument('--dataset', type=str, default='ogbl-collab')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")
# GNN settings
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--sortpool_k', type=float, default=0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=512)
# Subgraph extraction settings
parser.add_argument('--num_hops', type=int, default=3)
parser.add_argument('--ratio_per_hop', type=float, default=1.0)
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='drnl',
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true',
                    help="whether to use raw node features as GNN input")
# parser.add_argument('--use_edge_weight', action='store_true',
#                     help="whether to consider edge weight in GNN")
parser.add_argument('--use_edge_weight', default=True,
                    help="whether to consider edge weight in GNN")
# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
# parser.add_argument('--train_percent', type=float, default=100)
# parser.add_argument('--val_percent', type=float, default=100)
# parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--train_percent', type=float, default=1)
parser.add_argument('--val_percent', type=float, default=1)
parser.add_argument('--test_percent', type=float, default=1)
# parser.add_argument('--dynamic_train', action='store_true',
#                     help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_train', default=True,
                    help="dynamically extract enclosing subgraphs on the fly")
# parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_val', default=True,
                    help="dynamically extract enclosing subgraphs on the fly")
# parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--dynamic_test', default=True,
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--num_workers', type=int, default=16,
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true',
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--pretrained_node_embedding', type=str, default=None,
                    help="load pretrained node embeddings as additional node features")
# Testing settings
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--eval_steps', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--data_appendix', type=str, default='',
                    help="an appendix to the data directory")
parser.add_argument('--save_appendix', type=str, default='',
                    help="an appendix to the save directory")
parser.add_argument('--keep_old', action='store_true',
                    help="do not overwrite old files in the save directory")
parser.add_argument('--continue_from', type=int, default=None,
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--only_test', action='store_true',
                    help="only test without training")
parser.add_argument('--test_multiple_models', action='store_true',
                    help="test multiple models together")
parser.add_argument('--use_heuristic', type=str, default=None,
                    help="test a link prediction heuristic (CN or AA)")
parser.add_argument('--max_node_degree', type=int, default=5)
args = parser.parse_args()

if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
if args.data_appendix == '':
    args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)
    if args.use_valedges_as_input:
        args.data_appendix += '_uvai'

args.res_dir = os.path.join('results/{}{}'.format(args.dataset, args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
if not args.keep_old:
    # Backup python files.
    copy('seal_link_pred.py', args.res_dir)
    copy('utils.py', args.res_dir)
log_file = os.path.join(args.res_dir, 'log.txt')
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')
with open(log_file, 'a') as f:
    f.write('\n' + cmd_input)

if args.dataset.startswith('ogbl'):
    dataset = PygLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    if args.dataset.startswith('ogbl-vessel'):
        # normalize node features
        data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
        data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
        data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
else:
    path = osp.join('dataset', args.dataset)
    dataset = Planetoid(path, args.dataset)
    split_edge = do_edge_split(dataset, args.fast_split)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()

if args.dataset.startswith('ogbl-citation'):
    args.eval_metric = 'mrr'
    directed = True
elif args.dataset.startswith('ogbl-vessel'):
    args.eval_metric = 'rocauc'
    directed = False
elif args.dataset.startswith('ogbl'):
    args.eval_metric = 'hits'
    directed = False
else:  # assume other datasets are undirected
    args.eval_metric = 'auc'
    directed = False

if args.use_valedges_as_input:
    val_edge_index = split_edge['valid']['edge'].t()
    if not directed:
        val_edge_index = to_undirected(val_edge_index)
    data.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=int)
    data.edge_weight = torch.cat([data.edge_weight, val_edge_weight], 0)

if args.dataset.startswith('ogbl'):
    evaluator = Evaluator(name=args.dataset)
if args.eval_metric == 'hits':
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
elif args.eval_metric == 'mrr':
    loggers = {
        'MRR': Logger(args.runs, args),
    }
elif args.eval_metric == 'rocauc':
    loggers = {
        'rocauc': Logger(args.runs, args),
    }

elif args.eval_metric == 'auc':
    loggers = {
        'AUC': Logger(args.runs, args),
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.use_heuristic:
    # Test link prediction heuristics.
    num_nodes = data.num_nodes
    if 'edge_weight' in data:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])),
                       shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge,
                                                   data.edge_index,
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge,
                                                     data.edge_index,
                                                     data.num_nodes)
    pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
    neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
    pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'rocauc':
        results = evaluate_ogb_rocauc(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        val_pred = torch.cat([pos_val_pred, neg_val_pred])
        val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int),
                              torch.zeros(neg_val_pred.size(0), dtype=int)])
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int),
                               torch.zeros(neg_test_pred.size(0), dtype=int)])
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    for key, result in results.items():
        loggers[key].add_result(0, result)
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    pdb.set_trace()
    exit()

# SEAL.
path = dataset.root + '_seal{}'.format(args.data_appendix)
use_coalesce = True if args.dataset == 'ogbl-collab' else False
if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
    args.num_workers = 0

dataset_class = 'SEALDynamicDataset' if args.dynamic_train else 'SEALDataset'
train_dataset = eval(dataset_class)(
    path,
    data,
    split_edge,
    num_hops=args.num_hops,
    percent=args.train_percent,
    split='train',
    use_coalesce=use_coalesce,
    node_label=args.node_label,
    ratio_per_hop=args.ratio_per_hop,
    max_nodes_per_hop=args.max_nodes_per_hop,
    directed=directed,
)
if False:  # visualize some graphs
    import networkx as nx
    from torch_geometric.utils import to_networkx
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for g in loader:
        f = plt.figure(figsize=(20, 20))
        limits = plt.axis('off')
        g = g.to(device)
        node_size = 100
        with_labels = True
        G = to_networkx(g, node_attrs=['z'])
        labels = {i: G.nodes[i]['z'] for i in range(len(G))}
        nx.draw(G, node_size=node_size, arrows=True, with_labels=with_labels,
                labels=labels)
        f.savefig('tmp_vis.png')
        pdb.set_trace()

dataset_class = 'SEALDynamicDataset' if args.dynamic_val else 'SEALDataset'
val_dataset = eval(dataset_class)(
    path,
    data,
    split_edge,
    num_hops=args.num_hops,
    percent=args.val_percent,
    split='valid',
    use_coalesce=use_coalesce,
    node_label=args.node_label,
    ratio_per_hop=args.ratio_per_hop,
    max_nodes_per_hop=args.max_nodes_per_hop,
    directed=directed,
)
dataset_class = 'SEALDynamicDataset' if args.dynamic_test else 'SEALDataset'
test_dataset = eval(dataset_class)(
    path,
    data,
    split_edge,
    num_hops=args.num_hops,
    percent=args.test_percent,
    split='test',
    use_coalesce=use_coalesce,
    node_label=args.node_label,
    ratio_per_hop=args.ratio_per_hop,
    max_nodes_per_hop=args.max_nodes_per_hop,
    directed=directed,
)

max_z = 1000  # set a large max_z so that every z has embeddings to look up

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                         num_workers=args.num_workers)

if args.train_node_embedding:
    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
elif args.pretrained_node_embedding:
    weight = torch.load(args.pretrained_node_embedding)
    emb = torch.nn.Embedding.from_pretrained(weight)
    emb.weight.requires_grad = False
else:
    emb = None

if __name__ == "__main__":
    for run in range(args.runs):
        if args.model == 'DGCNN':
            model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k,
                          train_dataset, args.dynamic_train, use_feature=args.use_feature,
                          node_embedding=emb).to(device)
        elif args.model == 'SAGE':
            model = SAGE(args.hidden_channels, args.num_layers, max_z, train_dataset,
                         args.use_feature, node_embedding=emb).to(device)
        elif args.model == 'GCN':
            model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset,
                        args.use_feature, node_embedding=emb).to(device)
        elif args.model == 'GIN':
            model = GIN(args.hidden_channels, args.num_layers, max_z, train_dataset,
                        args.use_feature, node_embedding=emb).to(device)
        parameters = list(model.parameters())
        if args.train_node_embedding:
            torch.nn.init.xavier_uniform_(emb.weight)
            parameters += list(emb.parameters())
        # optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
        from optimizer.dp_optimizer import DPSGD

        learning_rate = 0.5
        numEpoch = 15000
        sigma = 1.23
        momentum = 0.9
        delta = 10 ** (-5)
        max_norm = 0.1
        max_terms_per_node = 20
        target_delta = 1e-5
        noise_multiplier = sigma * compute_base_sensitivity(max_degree=args.max_node_degree,
                                                            num_message_passing_steps=args.num_layers)
        optimizer = DPSGD(
            l2_norm_clip=max_norm,  # 裁剪范数
            noise_multiplier=noise_multiplier,
            minibatch_size=args.batch_size,  # 几个样本梯度进行一次梯度下降
            microbatch_size=1,  # 几个样本梯度进行一次裁剪，这里选择逐样本裁剪
            params=model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')
        if args.model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}')
        with open(log_file, 'a') as f:
            print(f'Total number of parameters is {total_params}', file=f)
            if args.model == 'DGCNN':
                print(f'SortPooling k is set to {model.k}', file=f)

        start_epoch = 1
        if args.continue_from is not None:
            model.load_state_dict(
                torch.load(os.path.join(args.res_dir,
                                        'run{}_model_checkpoint{}.pth'.format(run + 1, args.continue_from)))
            )
            optimizer.load_state_dict(
                torch.load(os.path.join(args.res_dir,
                                        'run{}_optimizer_checkpoint{}.pth'.format(run + 1, args.continue_from)))
            )
            start_epoch = args.continue_from + 1
            args.epochs -= args.continue_from

        if args.only_test:
            results = test()
            for key, result in results.items():
                loggers[key].add_result(run, result)
            for key, result in results.items():
                valid_res, test_res = result
                print(key)
                print(f'Run: {run + 1:02d}, '
                      f'Valid: {100 * valid_res:.2f}%, '
                      f'Test: {100 * test_res:.2f}%')
            pdb.set_trace()
            exit()

        if args.test_multiple_models:
            model_paths = [
            ]  # enter all your pretrained .pth model paths here
            models = []
            for path in model_paths:
                m = cp.deepcopy(model)
                m.load_state_dict(torch.load(path))
                models.append(m)
            Results = test_multiple_models(models)
            for i, path in enumerate(model_paths):
                print(path)
                with open(log_file, 'a') as f:
                    print(path, file=f)
                results = Results[i]
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (f'Run: {run + 1:02d}, ' +
                                f'Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')
                    print(key)
                    print(to_print)
                    with open(log_file, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)
            pdb.set_trace()
            exit()

        # Training starts
        for epoch in range(start_epoch, start_epoch + args.epochs):
            loss = train()

            if epoch % args.eval_steps == 0:
                results = test()
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    model_name = os.path.join(
                        args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run + 1, epoch))
                    optimizer_name = os.path.join(
                        args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run + 1, epoch))
                    torch.save(model.state_dict(), model_name)
                    torch.save(optimizer.state_dict(), optimizer_name)

                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                    f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    print(f'Total number of parameters is {total_params}')
    print(f'Results are saved in {args.res_dir}')
