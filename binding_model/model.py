import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINConv, GATConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data

class HybridGNN(nn.Module):
    def __init__(self, hidden_channels, descriptor_size, dropout_rate):
        super(HybridGNN, self).__init__()

        self.conv1 = GINConv(nn.Linear(9, hidden_channels))
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)

        combined_size = hidden_channels * 2 + descriptor_size
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.affinity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x, edge_index, batch_index, descriptors):
        h = self.gelu(self.conv1(x, edge_index))
        h = self.gelu(self.conv2(h, edge_index))
        h = self.gelu(self.conv3(h, edge_index))

        h = torch.cat([
            global_max_pool(h, batch_index),
            global_mean_pool(h, batch_index)
        ], dim=1)

        combined = torch.cat([h, descriptors], dim=1)
        shared_features = self.shared_fc(combined)
        affinity = self.affinity_head(shared_features)

        return affinity