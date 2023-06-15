import numpy as np
import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing

class TADW(MessagePassing):
    """
    Text Attributed DeepWalk Abstract Class
    """
    def __init__(self, M, T, k, lambd, lr, num_class, name):
        """
        Setting up the target matrices and arguments. Weights are initialized.
        :param A: Proximity matrix.
        :param T: Text data
        :param args: Model arguments.
        """
        super().__init__()

        self.M = M  # |V|*|V|
        self.T = T  # ft*|V|
        self.k = k
        self.lambd = lambd
        self.lr = lr
        self.name = name

        self.linearsvm = tlx.layers.Linear(in_features=2*k, out_features=num_class,
                                            W_init='xavier_uniform', b_init=None)

    def calcu_embed(self, max_iter):
        W = np.mat(np.random.random((self.M.shape[0], self.k)))  # |V|*k
        H = np.mat(np.random.random((self.T.shape[0], self.k)))  # ft*k

        _iter = 0
        loss_list = []
        while _iter < max_iter:
            # alternately minimize W and H
            W = W - self.lr * (2 * (W @ H.transpose() @ self.T - self.M) @ self.T.transpose() @ H + self.lambd * W)
            temp = self.T @ self.T.transpose() @ H @ W.transpose() @ W - self.T @ self.M.transpose() @ W
            H = H - self.lr * (2 * temp + self.lambd * H)
            loss = np.linalg.norm(self.M - W @ H.transpose() @ self.T, ord=2)
            print("iter: ", _iter, ", loss: ", loss)
            loss_list.append(loss)
            if loss <= 1e-3:
                break
            _iter += 1
        embedding = np.hstack((W, self.T.transpose() @ H * 10))  # |V|*2k
        # embedding = np.concatenate((W.transpose(), H.transpose() @ self.T * 10), axis=0)  # 2k*|V|
        return embedding  # |V|*2k

    # LinearSVM
    def forward(self, x):
        return self.linearsvm(x)
