import torch
import torch.nn as nn
from src.gat import GAT
from src.loss_func import sce_loss, ce_loss
from functools import partial
import dgl
from dgl import to_networkx
from src.utils import create_norm


import os, sys, time
# from src import *
from src.functional import community_detection, transition, community_strength, get_edge_weight, cav, ced
import numpy as np
from torch_geometric.utils import add_self_loops, negative_sampling, degree
# def create_norm(name):
#     if name == "layernorm":
#         return nn.LayerNorm
#     elif name == "batchnorm":
#         return nn.BatchNorm1d
#     # elif name == "graphnorm":
#     #     return partial(NormLayer, norm_type="groupnorm")
#     else:
#         return nn.Identity
def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")
class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for _, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x




class ae(nn.Module):
    def __init__(self, g):
        super(ae, self).__init__()
        self.norm = None
        self.encoder1 = GAT(
            in_dim=1433,
            num_hidden= 128,
            out_dim=128,
            num_layers=2,
            nhead=4,
            nhead_out=4,
            activation="prelu",
            feat_drop=0.2,
            attn_drop=0.1,
            negative_slope=0.2,
            residual=False,
            norm=create_norm(self.norm),
            concat_out=True,
            encoding=True,
        )
        self.encoder2 = GAT(
            in_dim=512,
            num_hidden=512,
            out_dim=512,
            num_layers=1,
            nhead=4,
            nhead_out=1,
            activation="prelu",
            feat_drop=0.2,
            attn_drop=0.1,
            negative_slope=0.2,
            residual=False,
            norm=create_norm(self.norm),
            concat_out=True,
            encoding=True,
        )
        self.decoder = GAT(
            in_dim=512,
            num_hidden= 512,
            out_dim=1433,
            num_layers=1,
            nhead=4,
            nhead_out=1,
            activation="prelu",
            feat_drop=0.2,
            attn_drop=0.1,
            negative_slope=0.2,
            residual=False,
            norm=create_norm(self.norm),
            concat_out=True,
            encoding=False,
        )
        
        self.encoder_to_decoder = nn.Linear(512, 512, bias=False)
        self._replace_rate = 0.2
        self.p1 = 0.6
        self.p2 = 0.7
        self.enc_mask_token = nn.Parameter(torch.zeros(1, 1433)).to("cuda")
        self.edge_index = torch.cat((g.edges()[0], g.edges()[1]), 0).view(2, -1) 
        self.edge_weight, self.node_cs = self.com_det(g)
        self.num_nodes = g.num_nodes()
        self.edge_decoder = EdgeDecoder(512, 215,num_layers=2, dropout=0.2)
        self.negative_sampler = negative_sampling

    def sig(self, f):
        return 0#f if f<0 else 0

    def com_det(self, graph):
        p1 = './log/par/'+'cora_'+'edge_weight.pt'
        p2 = "./log/par/"+'cora_'+'node_cs'
        if os.path.isfile(p1) and os.path.isfile(p2+'.npy'):
            edge_weight = torch.load(p1)
            node_cs = np.load(p2+'.npy')
        else:
        # edge_index = torch.cat((graph.edges()[0], graph.edges()[1]), 0).view(2, -1) 
            
            print('Detecting communities...')
            g = to_networkx(graph.to("cpu"))
            communities = community_detection('leiden')(g).communities
            com = transition(communities, g.number_of_nodes())
            com_cs, node_cs = community_strength(g, communities)
            edge_weight = get_edge_weight(self.edge_index, com, com_cs)
            com_size = [len(c) for c in communities]
            print(f'Done! {len(com_size)} communities detected. \n')
        
            torch.save(edge_weight, p1)
            np.save(p2, node_cs)
        return edge_weight, node_cs
        
    def forward(self, g, x):
        u_g, u_x,  mask_nodes, masked_edges = self.make_attr(g, x)
        rep = self.encoder1(u_g, u_x)
        # rep = self.encoder2(u_g, rep)

        # rep[mask_nodes] = 0
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0
        rep1 =self.decoder(u_g, rep)
        criterion = partial(sce_loss, alpha=3)
        # criterion = nn.MSELoss()
        loss = criterion(x[mask_nodes], rep1[mask_nodes])
        #******************* loss for edge prediction *********************
        aug_edge_index, _ = add_self_loops(self.edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=self.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        pos_out = self.edge_decoder(rep, masked_edges, sigmoid=False)
        neg_out = self.edge_decoder(rep, neg_edges, sigmoid=False)

        criterion0 = ce_loss
        ce_los = criterion0(pos_out, neg_out)
        ce_los = self.sig(ce_los)
        #******************************************************************

        return loss, ce_los
        
    def make_attr(self, g, x):

        u_g, u_x,  mask_nodes, mask_edge = self.encoding_mask_noise(g, x)
        # u_g, u_x,  mask_nodes, mask_edge = self.make_noise(x)
        return u_g, u_x,  mask_nodes, mask_edge


    def set_para(self, p1, p2, r_p):
        self.p1 = p1
        self.p2 = p2
        self._replace_rate = r_p


#best p1 = 0.1, p2 = 0.5
    def make_noise(self, x, p1 = 0.2, p2 = 0.4):

        p1 = self.p1
        p2 = self.p2

        # edge_index = self.edge_index#torch.cat((g.edges()[0], g.edges()[1]), 0).view(2, -1)
        # print(len(self.edge_index[0]))
        edge_index_1, mask_edge = ced(self.edge_index, self.edge_weight, p=p1, threshold=1.)
        # print(len(edge_index_1[0]))
        u_x, mask_nodes = cav(x, self.node_cs, p2, max_threshold=1.)

        use_g = dgl.graph((edge_index_1[0],edge_index_1[1]), num_nodes = self.num_nodes)
        use_g = use_g.remove_self_loop()
        use_g = use_g.add_self_loop()

        mask_nodes = torch.arange(self.num_nodes).to("cuda")[mask_nodes]
        num_nodes = self.num_nodes
        # perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = len(mask_nodes)
        num_noise_nodes = int(self._replace_rate * num_mask_nodes)
        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        token_nodes = mask_nodes[perm_mask[: int((1 - self._replace_rate) * num_mask_nodes)]]
        noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        # u_x[token_nodes] = 0.0
        u_x[noise_nodes] = x[noise_to_be_chosen]
        u_x[token_nodes] += self.enc_mask_token
        return use_g, u_x, mask_nodes, mask_edge

    def embed(self, graph, x):
        return self.encoder1(graph, x)
    
    
    def encoding_mask_noise(self, g, x, mask_rate=0.5):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        _, mask_edge = ced(self.edge_index, self.edge_weight, p=self.p1, threshold=1.)
        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int((1 - self._replace_rate) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, mask_nodes, mask_edge