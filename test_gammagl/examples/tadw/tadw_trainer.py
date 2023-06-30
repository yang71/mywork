import os
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'mindspore'
import sys
sys.path.insert(0, os.path.abspath('../../')) # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
import numpy as np
import pandas as pd
from gammagl.datasets import Planetoid
from gammagl.layers.conv import TADW
from gammagl.utils import add_self_loops, mask_to_index
from gammagl.utils import degree
from tensorlayerx.model import TrainOneStep, WithLoss


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'])
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss


def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """

    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def normalize_adjacency(edge_index, num_nodes):
    """
    Method to calculate a degree normalized adjacency matrix.
    :param graph: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    src = edge_index[0]
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
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)

    # M = (A + A * A)/2
    A = tlx.convert_to_numpy(tlx.convert_to_tensor(normalize_adjacency(edge_index, graph.num_nodes)))
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
    U, S, V = np.linalg.svd(feature)
    U_ft, S_ft = U[:, :ft], np.diag(S[:ft])
    text_feature = np.dot(U_ft, S_ft)  # |V|*ft
    length = len(text_feature)
    for i in range(length):
        temp = np.linalg.norm(text_feature[i], ord=2)
        if temp > 0:
            text_feature[i] = text_feature[i] / temp
    text_feature = text_feature.transpose() * 0.1

    k = 80  # 80
    lambd = 0.2  # 0.2 - 0.5
    lr = 0.001  # 0.001
    max_iter = 200  # 200
    model = TADW(M, text_feature, k=k, lambd=lambd, lr=lr, num_class=7, name='TADW')
    # k=80 lambd=0.5 lr=0.001 max_iter=200 --> loss=4
    # k=80 lambd=0.2 lr=0.001 max_iter=200 --> loss=2
    # embedding = model.calcu_embed(max_iter=max_iter)

    # print("\nSaving the embedding.\n")
    # columns = ["X_" + str(dim) for dim in range(2 * k)]
    # out = pd.DataFrame(embedding, columns=columns)
    # out.to_csv('embedding.csv', index=False)
    #
    embedding = pd.read_csv('embedding.csv').to_numpy()
    # length = len(embedding)
    # for i in range(length):
    #     temp = np.linalg.norm(embedding[i], ord=2)
    #     if temp > 0:
    #         embedding[i] = embedding[i] / temp
    # print(type(embedding))

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = model.trainable_weights

    loss_func = SemiSpvzLoss(model, tlx.losses.softmax_cross_entropy_with_logits)
    # loss_func = SemiSpvzLoss(model, tlx.losses.mean_squared_error)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    data = {
        "x": tlx.convert_to_tensor(embedding.astype(np.float32)),
        "y": graph.y,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx
    }

    # 训练模型
    num_epochs = 500
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.set_train()
        train_loss = train_one_step(data, data['y'])
        model.set_eval()
        logits = model(data['x'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc))

        # save best model on evaluation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

    model.load_weights(args.best_model_path + model.name + ".npz", format='npz_dict')

    model.set_eval()
    logits = model(data['x'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)
    print("Test acc:  {:.4f}".format(test_acc))


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
