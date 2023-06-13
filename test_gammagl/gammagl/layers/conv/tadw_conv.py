import numpy as np
from tqdm import tqdm

class TADW(object):
    """
    Text Attributed DeepWalk Abstract Class
    """
    def __init__(self, M, T, k, lambd, lr):
        """
        Setting up the target matrices and arguments. Weights are initialized.
        :param A: Proximity matrix.
        :param T: Text data
        :param args: Model arguments.
        """
        self.M = M  # |V|*|V|
        self.T = np.transpose(T)  # |V|*ft -> ft*|V|
        self.k = k
        self.lambd = lambd
        self.lr = lr
        self.init_weights()

    def init_weights(self):
        """
        Initialization of weights and loss container.
        """
        self.W = np.random.uniform(0, 1, (self.k, self.M.shape[0]))  # k*|V|
        self.H = np.random.uniform(0, 1, (self.k, self.T.shape[0]))  # k*ft

    def update_W(self):
        """
        A single update of the node embedding matrix.
        """
        H_T = np.dot(self.H, self.T)  # k*|V|
        grad = self.lambd * self.W - np.dot(H_T, self.M - np.dot(np.transpose(H_T), self.W))
        self.W = self.W - self.lr * grad
        # Overflow control
        # self.W[self.W < self.lower_control] = self.lower_control

    def update_H(self):
        """
        A single update of the feature basis matrix.
        """
        inside = self.M - np.dot(np.dot(np.transpose(self.W), self.H), self.T)  # |V|*|V|
        grad = self.lambd * self.H - np.dot(np.dot(self.W, inside), np.transpose(self.T))  # K*ft
        self.H = self.H - self.lr * grad
        # Overflow control
        # self.H[self.H < self.lower_control] = self.lower_control

    def calculate_loss(self, iteration):
        """
        Calculating the losses in a given iteration.
        :param iteration: Iteration round number.
        """
        main_loss = np.sum(np.square(self.M - np.dot(np.dot(np.transpose(self.W), self.H), self.T)))
        regul_1 = self.lambd * np.sum(np.square(self.W))
        regul_2 = self.lambd * np.sum(np.square(self.H))
        print(iteration, main_loss, regul_1, regul_2)

    def compile_embedding(self, ids):
        """
        Return the embedding.
        """
        to_concat = [ids, np.transpose(self.W), np.transpose(np.dot(self.H, self.T))]
        return np.concatenate(to_concat, axis=1)


    def optimize(self):
        """
        Gradient descent updates for a given number of iterations.
        """
        self.calculate_loss(0)
        for i in tqdm(range(1, 201)):
            self.update_W()
            self.update_H()
            self.calculate_loss(i)

    def solve(self, max_iter):
        W = np.mat(np.random.random((self.M.shape[0], self.k)))  # |V|*k
        H = np.mat(np.random.random((self.T.shape[0], self.k)))  # ft*k

        _iter = 0
        loss_list = []
        while _iter < max_iter:
            print(_iter)

            W = W - self.lr * (2 * (W * H.transpose() * self.T - self.M) * self.T.transpose() * H + self.lambd * W)

            # 这里报错！！！ 因为维度不对，这一步应该是在求解grad，但是不是很明白交替最小化的grad具体怎么求解
            temp = self.T * self.T.transpose() * H * W.transpose() * W - self.T * self.M.transpose() * W
            H = H - self.lr * (2 * temp + self.lambd * H)

            loss = np.linalg.norm(self.M - W * H.transpose() * self.T, ord=2)
            print(loss)
            loss_list.append(loss)
            if loss <= 1e-3:
                break
            _iter += 1

        W = np.hstack((W, self.T.transpose() * H * 10))
        return loss_list, W, H
