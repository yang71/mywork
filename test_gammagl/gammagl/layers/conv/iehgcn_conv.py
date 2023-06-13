import tensorlayerx as tlx
from gammagl.layers.conv import MessagePassing, GCNConv
from tensorlayerx.nn import ModuleDict, Linear, Dropout
from tensorlayerx import elu

class ieHGCNConv(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels,
            attn_channels,
            metadata,
            batchnorm=False,
            add_bias=False,
            activation=elu,
            dropout_rate=0.0):
        super(ieHGCNConv, self).__init__()

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_channels = attn_channels
        self.metadata = metadata
        self.batchnorm = batchnorm
        self.add_bias = add_bias
        self.dropout_rate = dropout_rate

        self.W_self = ModuleDict()
        self.W_al = ModuleDict()
        self.W_ar = ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.W_self[node_type] = Linear(in_features=in_channels, out_features=self.out_channels,
                                            W_init='xavier_uniform', b_init=None)
            self.W_al[node_type] = Linear(in_features=self.attn_channels, out_features=self.out_channels,
                                          W_init='xavier_uniform', b_init=None)
            self.W_ar[node_type] = Linear(in_features=self.attn_channels, out_features=self.out_channels,
                                          W_init='xavier_uniform', b_init=None)

        self.gcn_dict = ModuleDict({})
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            self.gcn_dict[edge_type] = GCNConv(in_channels=self.in_channels[src_type],
                                               out_channels=self.out_channels,
                                               norm='right')

        self.linear_q = ModuleDict()
        self.linear_k = ModuleDict()
        for node_type, _ in self.in_channels.items():
            self.linear_q[node_type] = Linear(in_features=self.out_channels, out_features=self.attn_channels,
                                              W_init='xavier_uniform', b_init=None)
            self.linear_k[node_type] = Linear(in_features=self.out_channels, out_features=self.attn_channels,
                                              W_init='xavier_uniform', b_init=None)

        self.activation = activation

        if self.batchnorm:
            self.bn = tlx.layers.BatchNorm1d(num_features=out_channels)
        if self.add_bias:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1, self.out_channels), init=initor)

        self.dropout = Dropout(self.dropout_rate)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()

        dst_dict, out_dict = {}, {}

        # formulas (2)-1
        # Iterate over node-types:
        for node_type, x in x_dict.items():
            dst_dict[node_type] = self.W_self[node_type](x)  # paper: 6*5 author : 4*5
            out_dict[node_type] = []

        query = {}
        key = {}
        attn = {}
        attention = {}

        # formulas (3)-1 and (3)-2
        for node_type, _ in x_dict.items():
            query[node_type] = self.linear_q[node_type](dst_dict[node_type])  # q: paper: 6*32 author: 4*32
            key[node_type] = self.linear_k[node_type](dst_dict[node_type])  # k-self : paper: 6*32 author: 4*32

        # formulas (4)-1
        h_l = {}
        h_r = {}
        for node_type, _ in x_dict.items():
            h_l[node_type] = self.W_al[node_type](key[node_type])  # 6*5  4*5
            h_r[node_type] = self.W_ar[node_type](query[node_type])

        for node_type, x in x_dict.items():
            attention[node_type] = elu(h_l[node_type] + h_r[node_type])
            attention[node_type] = tlx.expand_dims(attention[node_type], axis=0)  # author: 1*4*5  paper: 1*6*5

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            # formulas (2)-2
            out = self.gcn_dict[edge_type](x_dict[src_type], edge_index, num_nodes=num_nodes_dict[dst_type])
            out_dict[dst_type].append(out)

            # formulas (3)-3
            attn[dst_type] = self.linear_k[dst_type](out)
            # formulas (4)-2
            h_attn = self.W_al[dst_type](attn[dst_type])
            attn.clear()

            edge_attention = elu(h_attn + h_r[dst_type])
            edge_attention = tlx.expand_dims(edge_attention, axis=0)
            attention[dst_type] = tlx.concat([attention[dst_type], edge_attention], axis=0)

        # formulas (5)
        for node_type, _ in x_dict.items():
            attention[node_type] = tlx.softmax(attention[node_type], axis=0)  # 2*4*5 2*6*5

        # formulas (6)
        rst = {node_type: 0 for node_type, _ in x_dict.items()}
        for node_type, data in out_dict.items():
            data = [dst_dict[node_type]] + data  # dis_dict是经过W-self的Z-self, data是经过gcn的Z-neighbor
            if len(data) != 0:
                for i in range(len(data)):
                    aggregation = tlx.multiply(data[i], attention[node_type][i])
                    rst[node_type] = aggregation + rst[node_type]

        def _apply(ntype, h):
            if self.add_bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)

        # profiler.stop()
        # profiler.open_in_browser()

        return {ntype: _apply(ntype, feat) for ntype, feat in rst.items()}

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

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))

    conv = ieHGCNConv(
        in_channels=16,
        out_channels=5,
        attn_channels=32,
        metadata=metadata,
        batchnorm=True,
        add_bias=True,
        activation=elu,
        dropout_rate=0.5)
    num_nodes_dict = {'author': 4, 'paper': 6}
    out_dict = conv(x_dict, edge_index_dict, num_nodes_dict)
    print(out_dict)