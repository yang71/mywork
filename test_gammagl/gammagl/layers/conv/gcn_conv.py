import tensorlayerx as tlx
from gammagl.utils import degree
from gammagl.layers.conv import MessagePassing


class GCNConv(MessagePassing):

    def __init__(self, 
                in_channels,
                out_channels,
                norm='both',
                add_bias=True):
        super().__init__()
        
        if norm not in ['left', 'right', 'none', 'both']:
            raise ValueError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                             ' But got "{}".'.format(norm))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self._norm = norm

        self.linear = tlx.layers.Linear(out_features=out_channels,
                                        in_features=in_channels,
                                        W_init='xavier_uniform',
                                        b_init=None)
        if add_bias is True:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, self.out_channels), init=initor)

    def forward(self, x, edge_index, edge_weight=None, num_nodes=None):

        x = self.linear(x)
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(edge_index.shape[1], 1))
        edge_weight = tlx.reshape(edge_weight,(-1,))
        weights = edge_weight
        
        if self._norm in ['left', 'both']:
            max_num = tlx.reduce_max(src)
            num_nodes = max_num + 1
            deg = degree(src, num_nodes=num_nodes, dtype = tlx.float32)
            if self._norm == 'both':
                norm = tlx.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))

        if self._norm in ['right', 'both']:
            max_num = tlx.reduce_max(dst)
            num_nodes = max_num + 1
            deg = degree(dst, num_nodes=num_nodes, dtype=tlx.float32)
            if self._norm == 'both':
                norm = tlx.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)

        out = self.propagate(x, edge_index, edge_weight=weights, num_nodes=num_nodes)
        if self.add_bias:
            out += self.bias
        return out


if __name__ == "__main__":
    x_dict = {
        'author': tlx.random_normal(shape=(4, 16)),
        'paper': tlx.random_normal(shape=(6, 16))
    }
    index1 = tlx.convert_to_tensor([0, 0, 1, 0, 3, 1, 0, 2, 0, 2, 0, 0, 2, 0, 3, 3, 1, 0, 2, 3], dtype=tlx.int32)
    index2 = tlx.convert_to_tensor([5, 3, 2, 3, 3, 1, 1, 5, 1, 5, 5, 0, 3, 2, 1, 5, 1, 5, 5, 1], dtype=tlx.int32)

    edge_index_dict = {
        ("author", "writes", "paper"): tlx.stack([index1, index2]),
        ("paper", "written_by", "author"): tlx.stack([index2, index1]),
    }

    conv = GCNConv(16, 3)
    num_nodes_dict = {'author': 4, 'paper': 6}
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        res = conv(x_dict[src_type], edge_index, num_nodes=num_nodes_dict[dst_type])

