import os

import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing
from gammagl.utils import add_self_loops

class GMMConv(MessagePassing):
    r"""The Gaussian Mixture Model Convolution or MoNet operator from the `"Geometric deep learning on graphs
    and manifolds using mixture model CNNs" <https://arxiv.org/abs/1611.08402>`_ paper

    .. math::
        u_{ij} &= f(x_i, x_j), x_j \in \mathcal{N}(i)

        w_k(u) &= \exp\left(-\frac{1}{2}(u-\mu_k)^T \Sigma_k^{-1} (u - \mu_k)\right)

        h_i^{l+1} &= \mathrm{aggregate}\left(\left\{\frac{1}{K}
         \sum_{k}^{K} w_k(u_{ij}), \forall j\in \mathcal{N}(i)\right\}\right)

    where :math:`u` denotes the pseudo-coordinates between a vertex and one of its neighbor,
    computed using function :math:`f`, :math:`\Sigma_k^{-1}` and :math:`\mu_k` are
    learnable parameters representing the covariance matrix and mean vector of a Gaussian kernel.

    Parameters
    ----------
    in_channels : int or tuple
        Size of each input sample, or :obj:`-1` to derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target dimensionalities.
    out_channels : int
        Size of each output sample.
    dim : int
        Dimensionality of pseudo-coordinte; i.e, the number of dimensions of :math:`u_{ij}`.
    n_kernels : int
        Number of kernels :math:`K`.
    aggregator_type : str
        Aggregator type (``sum``, ``mean``, ``max``). Default: ``sum``.
    add_bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_channels,  # 先不考虑异质图
                 out_channels,
                 dim,
                 n_kernels,
                 aggr='sum',
                 add_bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.n_kernels = n_kernels
        self.add_bias = add_bias
        if aggr == 'sum' or aggr == 'mean' or aggr == 'max':
            self.aggr = aggr
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggr))

        # 采用dgl初始化
        initor = tlx.initializers.RandomNormal(0, 0.1)
        self.mu = self._get_weights("mu", shape=(1, n_kernels, dim), init=initor, order='True')
        # self.mu = self._get_weights("mu", shape=(dim, n_kernels, 1), init=initor)  # torch用这个
        initor = tlx.initializers.Ones()
        self.sigma = self._get_weights("sigma", shape=(1, n_kernels, dim), init=initor, order='True')
        # self.sigma = self._get_weights("sigma", shape=(dim, n_kernels, 1), init=initor)  # torch用这个
        self.fc = tlx.layers.Linear(in_features=in_channels,
                                    out_features=n_kernels * out_channels,
                                    W_init='xavier_uniform',
                                    b_init=None)
        self.pseudo_fc = tlx.layers.Linear(in_features=dim,
                                    out_features=2,
                                    W_init='xavier_uniform')
        if self.add_bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, self.out_channels), init=initor)
    def forward(self, x, edge_index, pseudo=None):
        r"""

        Description
        -----------
        Compute Gaussian Mixture Model Convolution layer.

        Parameters
        ----------
        x : torch.Tensor
            The input node features.
        pseudo : torch.Tensor
            The pseudo coordinate tensor of shape :math:`(E, D_{u})` where
            :math:`E` is the number of edges of the graph and :math:`D_{u}`
            is the dimensionality of pseudo coordinate.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        src_feat, dst_feat = x, x
        num_nodes = int(dst_feat.shape[0])  # 2708
        E = edge_index.shape[1]  # 记录边的数目 13264
        src_feat = self.fc(src_feat)  # 2708*1433->2708*21(21=3*7)
        src_feat = tlx.reshape(src_feat, (-1, self.n_kernels, self.out_channels))  # 2708*3*7

        # 如下两步骤dgl中并没有，但是论文中提到了所以加上，影响也是不大的
        pseudo = self.pseudo_fc(pseudo)
        pseudo = tlx.tanh(pseudo)

        pseudo = tlx.reshape(pseudo, (E, 1, self.dim))  # 13264*2->13264*1*2
        # self.mu = tlx.reshape(self.mu, (1, self.n_kernels, self.dim))  # 1*3*2
        gaussian = -0.5 * ((pseudo - self.mu) ** 2)  # 13262*3*2
        # self.sigma = tlx.reshape(self.sigma, (1, self.n_kernels, self.dim))  #  1*3*2
        gaussian = gaussian * (self.sigma ** 2)  # 13264*3*2
        # gaussian = tlx.exp(tlx.reduce_sum(gaussian, -1, keepdims=True))  # (E, K, 1)
        gaussian = tlx.exp(tlx.reduce_sum(gaussian, -1))  # (E, K) 13264*3
        weights = gaussian  # 将其作为边权重
        out = self.propagate(src_feat, edge_index, edge_weight=weights, num_nodes=num_nodes, aggr=self.aggr)  # 2708*3*7?
        out = tlx.reduce_sum(out, -2)  # 2708*7
        if self.add_bias:
            out = out + self.bias
        return out


if __name__ == "__main__":
    edge_index = tlx.convert_to_tensor([[0, 1, 2, 3, 6, 2, 5],
                                        [1, 2, 3, 4, 2, 0, 3]], dtype=tlx.int64)
    # u, v = edge_index[0, :], edge_index[1, :]
    edge_index, _ = add_self_loops(edge_index)
    x = tlx.ones((7, 10))  # x就是feat
    pseudo = tlx.ones((14, 2))
    conv = GMMConv(10, 2, 2, 3, 'mean')
    res = conv(x, edge_index, pseudo)
    print(res)



