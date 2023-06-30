import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'tensorflow'  # default
# os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'mindspore'

import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import TADWModel

from sklearn import svm
from sklearn import metrics
from sklearn import model_selection


def calculate_acc(train_z, train_y, test_z, test_y):
    clf = svm.LinearSVC(C=5.0)
    clf.fit(train_z, train_y)
    predict_y = clf.predict(test_z)
    return metrics.accuracy_score(test_y, predict_y)


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index = graph.edge_index

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
        "num_nodes": graph.num_nodes,
    }

    best_test_acc = 0
    z_test = 0
    best_epoch = -1
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = model.fit()
        model.set_eval()
        z = model.compute()

        train_x, test_x, train_y, test_y = model_selection.train_test_split(z, tlx.convert_to_numpy(data['y']),
                                                                            test_size=0.5, shuffle=True)
        test_acc = calculate_acc(train_x, train_y, test_x, test_y)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            z_test = z
            best_epoch = epoch

        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  test acc: {:.4f}".format(test_acc))

    print("best_epoch: ", best_epoch)
    z = z_test

    train_x, test_x, train_y, test_y = model_selection.train_test_split(z, tlx.convert_to_numpy(graph.y),
                                                                        test_size=0.5, shuffle=True)
    test_acc = calculate_acc(train_x, train_y, test_x, test_y)
    print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    # parser.add_argument('--dataset', type=str, default='citeseer', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")  # 0.3
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epoch")  # 50 100
    parser.add_argument("--embedding_dim", type=int, default=80)  # 80 100 200 300 400 500
    parser.add_argument("--lamda", type=float, default=0.2)  # 0.5
    parser.add_argument("--svdft", type=int, default=200)  # 200 300

    args = parser.parse_args()

    # main(args)

    # old
    # test_acc = [0.7925, 0.8006, 0.8102, 0.8109, 0.8013]  # mindspore
    # test_acc = [0.8227, 0.8028, 0.8058, 0.8242, 0.7999]  # tensorflow
    # test_acc = [0.8006, 0.7829, 0.8072, 0.8198, 0.7777]  # torch
    # test_acc = [0.7954, 0.8013, 0.7939, 0.7843, 0.7767]  # paddle

    test_acc = []
    import numpy as np

    for i in range(5):
        test_acc.append(main(args))
    acc_mean = np.mean(test_acc)
    acc_std = np.std(test_acc)
    print("test_acc: ")
    print(test_acc)
    print("acc_mean: ", acc_mean, "acc_std: ", acc_std)