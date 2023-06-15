import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'tensorflow'

import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import TADWModel
from gammagl.utils import calc_gcn_norm, mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection


def calculate_acc(train_z, train_y, test_z, test_y, solver='lbfgs', multi_class='auto'):
    # train_z = tlx.convert_to_numpy(train_z)
    # train_y = tlx.convert_to_numpy(train_y)
    # test_z = tlx.convert_to_numpy(test_z)
    # test_y = tlx.convert_to_numpy(test_y)

    # clf = LogisticRegression(solver=solver, multi_class=multi_class).fit(train_z, train_y)
    # return clf.score(test_z, test_y)

    clf = svm.LinearSVC(C=5.0)
    clf.fit(train_z, train_y)
    predict_y = clf.predict(test_z)
    return metrics.accuracy_score(test_y, predict_y)


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'pubmed', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index = graph.edge_index

    # for mindspore, it should be passed into node indices
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    model = TADWModel(edge_index=edge_index,
                      embedding_dim=args.embedding_dim,
                      lr=args.lr,
                      lamda=args.lamda,
                      svdft=args.svdft,
                      node_feature=graph.x,
                      name="TADW")

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = model.fit(epoch)
        model.set_eval()
        z = model.campute()

        # val_acc = calculate_acc(tlx.gather(z, data['train_idx']), tlx.gather(graph.y, data['train_idx']),
        #                         tlx.gather(z, data['val_idx']), tlx.gather(graph.y, data['val_idx']))
        train_x, test_x, train_y, test_y = model_selection.train_test_split(z, tlx.convert_to_numpy(graph.y),
                                                                          test_size=0.2, shuffle=True)
        test_acc = calculate_acc(train_x, train_y, test_x, test_y)

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.10f}".format(train_loss.item()) \
              + "  val acc: {:.10f}".format(test_acc))

    model.set_eval()
    z = model.campute()
    train_x, test_x, train_y, test_y = model_selection.train_test_split(z, tlx.convert_to_numpy(graph.y),
                                                                        test_size=0.2, shuffle=True)
    test_acc = calculate_acc(train_x, train_y, test_x, test_y)
    print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--lr", type=float, default=0.3, help="learning rate")  # 0.3
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")  # 100
    parser.add_argument("--embedding_dim", type=int, default=500)  # 80 100 200 300 400
    parser.add_argument("--lamda", type=float, default=0.5)  # 0.5
    parser.add_argument("--svdft", type=int, default=300)  # 300

    args = parser.parse_args()

    main(args)
