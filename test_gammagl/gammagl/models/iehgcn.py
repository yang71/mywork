import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv import ieHGCNConv
from tensorlayerx import elu

class ieHGCNModel(tlx.nn.Module):
    def __init__(self,
                num_layers,
                in_channels,
                hidden_channels,
                out_channels,
                attn_channels,
                metadata,
                batchnorm=False,
                add_bias=False,
                activation=elu,
                dropout_rate=0.0,
                name=None,
                ):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.activation = elu

        if self.num_layers == 1:
            self.conv = nn.ModuleList([ieHGCNConv(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  attn_channels=attn_channels,
                                                  metadata=metadata,
                                                  batchnorm=batchnorm,
                                                  add_bias=add_bias,
                                                  activation=activation,
                                                  dropout_rate=dropout_rate)])
        else:
            self.conv = nn.ModuleList([ieHGCNConv(in_channels=in_channels,
                                                  out_channels=hidden_channels[0],
                                                  attn_channels=attn_channels,
                                                  metadata=metadata,
                                                  batchnorm=batchnorm,
                                                  add_bias=add_bias,
                                                  activation=activation,
                                                  dropout_rate=dropout_rate)])
            for i in range(1, num_layers - 1):
                self.conv.append(ieHGCNConv(in_channels=hidden_channels[i-1],
                                            out_channels=hidden_channels[i],
                                            attn_channels=attn_channels,
                                            metadata=metadata,
                                            batchnorm=batchnorm,
                                            add_bias=add_bias,
                                            activation=activation,
                                            dropout_rate=dropout_rate))
            self.conv.append(ieHGCNConv(in_channels=hidden_channels[-1],
                                        out_channels=out_channels,
                                        attn_channels=attn_channels,
                                        metadata=metadata,
                                        batchnorm=batchnorm,
                                        add_bias=add_bias,
                                        activation=activation,
                                        dropout_rate=dropout_rate))

    def forward(self, x_dict, edge_index_dict, num_nodes_dict):
        for i in range(self.num_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict, num_nodes_dict)
        return x_dict
