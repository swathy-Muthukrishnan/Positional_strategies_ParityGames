

import torch

from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GATConv 



class _ParityGameGATConv(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_attr, u=None, batch = None): 

        #x = self.conv1(x = x, edge_index =  edge_index, edge_attr =  edge_attr)
        #x = self.core(x, edge_index) # embeddings per node of shape : Number of nodes X Number of elements in learnable weight matrix
        x = self.encode(x = x, edge_index =  edge_index, edge_attr =  edge_attr, u= u)
        
        x = self.core(x, edge_index)
        #z = self.decode(x, edge_index)
        #u = u.reshape(-1, 6)
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        #if batch != None: 
        #    u = u[batch]
        
        edge_feat = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        
        #row, col = edge_index
        #edge_rep = torch.cat([x[row], x[col]], dim=1) # Edge representation is basically just the concatenation of the nodes involved in the edge. 

        return (self.node_classifier(x), self.edge_classifier(edge_feat))
        


class ParityGameGATConv(_ParityGameGATConv):

    def __init__(self, hidden_channels_nodes, hidden_channels_edges, core_iterations, config):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(in_channels = config['node_feat'], in_edge_channels = config['edge_feat'], out_channels = hidden_channels_nodes)
        self.conv2 = GATConv(128, hidden_channels_nodes)
        self.core = GAT(hidden_channels_nodes, hidden_channels_nodes, core_iterations, jk='lstm', flow='target_to_source')
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels_nodes, hidden_channels_nodes),  
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )

        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(((2 * hidden_channels_nodes) + config['edge_feat']) , hidden_channels_edges),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, config['classes']),
            torch.nn.Softmax(dim = 1)
        )
    
    def encode(self, x, edge_index, edge_attr, u, batch = None):
        #u = u.reshape(-1, 6)
        if batch != None: 
            u = u[batch]

        return self.conv1(x, edge_index, edge_attr = edge_attr).relu()
        #return self.conv2(x, edge_index).relu()

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()



