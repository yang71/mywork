import os
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'mindspore'
import sys

sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
import numpy as np
from gammagl.datasets import Planetoid
from gammagl.layers.conv import TADW
from gammagl.utils import add_self_loops
from gammagl.utils import degree


def normalize_adjacency(graph):
    """
    Method to calculate a degree normalized adjacency matrix.
    :param graph: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    num_nodes = graph.num_nodes
    src = graph.edge_index[0]
    degs = degree(src, num_nodes=num_nodes, dtype=tlx.float32)
    norm_degs = tlx.convert_to_numpy(1.0 / degs)
    # 初始化A
    A = [[0] * num_nodes for _ in range(num_nodes)]
    length = edge_index.shape[1]
    for i in range(length):
        src = edge_index[0][i]
        dst = edge_index[1][i]
        A[src][dst] = norm_degs[src]
    return A


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]

    # M = (A + A * A)/2
    A = tlx.convert_to_numpy(tlx.convert_to_tensor(normalize_adjacency(graph)))
    M = (A + np.dot(A, A)) / 2.  # |V|*|V|

    # TFIDF
    num_nodes = graph.num_nodes
    num_feats = graph.x.shape[1]
    feature = tlx.convert_to_numpy(graph.x)

    for i in range(num_feats):
        temp = tlx.convert_to_numpy(tlx.reduce_sum(feature[:, i]))
        if temp > 0:
            IDF = tlx.convert_to_numpy(tlx.log(num_nodes/temp))
            feature[:, i] = feature[:, i] * IDF
    # feature = tlx.convert_to_tensor(feature)

    # SVD
    ft = 200  # 只保留前k个奇异值对应的部分
    U, S, V = np.linalg.svd(feature, full_matrices=False)
    U_ft, S_ft = U[:, :ft], np.diag(S[:ft])
    text_feature = np.dot(U_ft, S_ft)  # |V|*ft

    k = 80
    lambd = 1000.0
    lr = 0.01
    model = TADW(M, text_feature, k=k, lambd=lambd, lr=lr)
    # 这个是第一个github的实现，跑通了，但是loss值输出不太对，还没找到原因
    # model.optimize()
    # 这个是第二个github的实现，还没跑通，grad的问题，具体在tadw_conv文件里面标注了
    loss, W, H = model.solve(max_iter=200)


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learnin rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="number of epoch")
    parser.add_argument("--hidden_dim", type=int, default=16, help="dimention of hidden layers")
    parser.add_argument("--drop_rate", type=float, default=0.5, help="drop_rate")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--norm", type=str, default='both', help="how to apply the normalizer.")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    args = parser.parse_args()

    main(args)
