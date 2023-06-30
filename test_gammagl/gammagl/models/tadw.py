import tensorlayerx as tlx
from collections import defaultdict
import numpy as np
from ..utils.num_nodes import maybe_num_nodes
from gammagl.utils import add_self_loops
from gammagl.utils import degree
from scipy.sparse.linalg import svds

lower_control = 10 ** (-15)

class TADWModel(tlx.nn.Module):
    def __init__(
            self,
            edge_index,
            embedding_dim,
            lr,
            lamda,
            svdft,
            node_feature,
            num_nodes=None,
            name=None
    ):
        super(TADWModel, self).__init__(name=name)

        self.edge_index = edge_index
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.lamda = lamda
        self.svdft = svdft
        self.node_feature = node_feature

        self.N = maybe_num_nodes(edge_index, num_nodes)

        self.M = self._create_target_matrix()
        self.T = self._create_tfifd_matrix()
        self.T = np.transpose(self.T)

        self.W = np.random.uniform(-1, 1, (self.embedding_dim, self.M.shape[0]))  # k*|V|
        self.H = np.random.uniform(-1, 1, (self.embedding_dim, self.T.shape[0]))  # k*ft

        self.W = self.W / np.linalg.norm(self.W)
        self.H = self.H / np.linalg.norm(self.H)

        # xavier_init也不能解决梯度爆炸的问题
        # def xavier_init(shape):
        #     n = shape[0]
        #     stddev = 1.0 / np.sqrt(n)
        #     return np.random.normal(loc=0, scale=stddev, size=shape)
        #
        # self.W = xavier_init((self.embedding_dim, self.M.shape[0]))  # k*|V|
        # self.H = xavier_init((self.embedding_dim, self.T.shape[0]))  # k*ft

        self.losses = []

    def _create_target_matrix(self):
        edge_index, _ = add_self_loops(self.edge_index, num_nodes=self.N, n_loops=1)
        num_nodes = self.N
        src = edge_index[0]
        degs = degree(src, num_nodes=num_nodes, dtype=tlx.float32)
        norm_degs = tlx.convert_to_numpy(1.0 / degs)

        A = [[0] * num_nodes for _ in range(num_nodes)]

        length = edge_index.shape[1]
        for i in range(length):
            src = edge_index[0][i]
            dst = edge_index[1][i]
            A[src][dst] = norm_degs[src]

        # graph_length = len(A)
        # for i in range(graph_length):
        #     if np.linalg.norm(A[i], ord=2) > 0:
        #         temp = tlx.convert_to_numpy(tlx.reduce_sum(tlx.convert_to_tensor(A[i])))
        #         A[i] = A[i] / temp

        M = (A + np.dot(A, A)) / 2.

        graph_length = len(M)
        for i in range(graph_length):
            if np.linalg.norm(M[i], ord=2) > 0:
                temp = tlx.convert_to_numpy(tlx.reduce_sum(tlx.convert_to_tensor(M[i])))
                M[i] = M[i] / temp

        return M

    def _create_tfifd_matrix(self):
        num_nodes = self.N
        num_feats = self.node_feature.shape[1]
        feature = tlx.convert_to_numpy(self.node_feature)

        for i in range(num_feats):
            temp = tlx.convert_to_numpy(tlx.reduce_sum(tlx.convert_to_tensor(feature[:, i])))
            if temp > 0:
                IDF = tlx.convert_to_numpy(tlx.log(tlx.convert_to_tensor(num_nodes / temp)))
                feature[:, i] = feature[:, i] * IDF

        U, S, V = svds(feature, k=self.svdft)
        text_feature = U.dot(np.diag(S))

        length = len(text_feature[0])
        for i in range(length):
            temp = np.linalg.norm(text_feature[:, i], ord=2)
            if temp > 0:
                text_feature[:, i] = text_feature[:, i] / temp

        # length = len(text_feature)
        # for i in range(length):
        #     np.linalg.norm(text_feature[i], ord=2)
        #     if temp > 0:
        #         text_feature[i] = text_feature[i] / temp
        # text_feature = text_feature * 0.1

        return text_feature

    def fit(self):
        """
        Gradient descent updates for a given number of iterations.
        """
        loss = self.loss()
        self.update_W()
        self.update_H()
        return loss

    def update_W(self):
        """
        A single update of the node embedding matrix.
        """

        H_T = np.dot(self.H, self.T)  # k*|V|
        grad = self.lamda * self.W - np.dot(H_T, self.M - np.dot(np.transpose(H_T), self.W))
        # grad = self.lamda * self.W - 2 / np.prod(self.M.shape) * np.dot(H_T, self.M - np.dot(H_T.T, self.W))
        self.W = self.W - self.lr * grad
        # Overflow control
        self.W = np.where(abs(self.W) < lower_control, np.sign(self.W) * lower_control, self.W)
        # rows = len(self.W)
        # cols = len(self.W[0])
        # for i in range(rows):
        #     for j in range(cols):
        #         if np.abs(self.W[i][j]) < lower_control:
        #             self.W[i][j] = lower_control * (self.W[i][j] / np.abs(self.W[i][j]))


    def update_H(self):
        """
        A single update of the feature basis matrix.
        """

        inside = self.M - np.dot(np.dot(np.transpose(self.W), self.H), self.T)  # |V|*|V|
        grad = self.lamda * self.H - np.dot(np.dot(self.W, inside), np.transpose(self.T))  # K*ft
        # grad = self.lamda * self.H - 2 / np.prod(self.M.shape) * np.dot(np.dot(self.W, inside), self.T.T)
        self.H = self.H - self.lr * grad
        # Overflow control
        self.H = np.where(abs(self.H) < lower_control, np.sign(self.H) * lower_control, self.H)
        # rows = len(self.H)
        # cols = len(self.H[0])
        # for i in range(rows):
        #     for j in range(cols):
        #         if np.abs(self.H[i][j]) < lower_control:
        #             self.H[i][j] = lower_control * (self.H[i][j] / np.abs(self.H[i][j]))

    def loss(self):
        # main_loss = np.mean(np.square(self.score_matrix))
        # regul_1 = self.lamda * np.mean(np.square(self.W)) / 2
        # regul_2 = self.lamda * np.mean(np.square(self.H)) / 2

        self.score_matrix = self.M - np.dot(np.dot(np.transpose(self.W), self.H), self.T)
        main_loss = np.sum(np.square(self.score_matrix))
        regul_1 = self.lamda * np.sum(np.square(self.W)) / 2
        regul_2 = self.lamda * np.sum(np.square(self.H)) / 2
        main_loss = main_loss + regul_1 + regul_2
        return main_loss

    def compute(self):
        feature = [np.transpose(self.W), np.transpose(np.dot(self.H, self.T))]
        feature = np.concatenate(feature, axis=1)
        length = len(feature[0])
        for i in range(length):
            temp = np.linalg.norm(feature[:, i], ord=2)
            if temp > 0:
                feature[:, i] = feature[:, i] / temp
        return feature