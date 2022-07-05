import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_max
from sklearn.preprocessing import normalize

class PosNet(nn.Module):
    def __init__(self, device):
        super(PosNet, self).__init__()
        self.device = device
        
        h = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 16, 3]

        self.conv1  = GCNConv(h[0], h[1])
        self.conv2  = GCNConv(h[1], h[2])
        self.conv3  = GCNConv(h[2], h[3])
        self.conv4  = GCNConv(h[3], h[4])
        self.conv5  = GCNConv(h[4], h[5])
        self.conv6  = GCNConv(h[5], h[6])
        self.conv7  = GCNConv(h[6], h[7])
        self.conv8  = GCNConv(h[7], h[8])
        self.conv9  = GCNConv(h[8], h[9])
        self.conv10 = GCNConv(h[9], h[10])
        self.conv11 = GCNConv(h[10], h[11])
        self.conv12 = GCNConv(h[11], h[12])

        self.linear1 = nn.Linear(h[12], h[13])
        self.linear2 = nn.Linear(h[13], h[14])
    
        self.bn1 = nn.BatchNorm1d(h[1])
        self.bn2 = nn.BatchNorm1d(h[2])
        self.bn3 = nn.BatchNorm1d(h[3])
        self.bn4 = nn.BatchNorm1d(h[4])
        self.bn5 = nn.BatchNorm1d(h[5])
        self.bn6 = nn.BatchNorm1d(h[6])
        self.bn7 = nn.BatchNorm1d(h[7])
        self.bn8 = nn.BatchNorm1d(h[8])
        self.bn9 = nn.BatchNorm1d(h[9])
        self.bn10 = nn.BatchNorm1d(h[10])
        self.bn11 = nn.BatchNorm1d(h[11])
        self.bn12 = nn.BatchNorm1d(h[12])

        self.l_relu = nn.LeakyReLU()
            

    def forward(self, data):

        z1, x_pos, edge_index = data.z1.to(self.device), data.x_pos.to(self.device), data.edge_index.to(self.device)
        n1 = torch.randn(x_pos.shape[0], x_pos.shape[1]).to(self.device) * 1e-5
        dx = self.l_relu(self.bn1(self.conv1(z1, edge_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, edge_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, edge_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, edge_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, edge_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, edge_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, edge_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, edge_index)))
        dx = self.l_relu(self.bn9(self.conv9(dx, edge_index)))
        dx = self.l_relu(self.bn10(self.conv10(dx, edge_index)))
        dx = self.l_relu(self.bn11(self.conv11(dx, edge_index)))
        dx = self.l_relu(self.bn12(self.conv12(dx, edge_index)))
        
        dx = self.l_relu(self.linear1(dx))
        dx = self.linear2(dx)
        
        return x_pos + dx

class NormalNet(nn.Module):
    def __init__(self, device):
        super(NormalNet, self).__init__()
        self.device = device
        
        h = [7, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 16, 3]

        self.conv1  = GCNConv(h[0], h[1])
        self.conv2  = GCNConv(h[1], h[2])
        self.conv3  = GCNConv(h[2], h[3])
        self.conv4  = GCNConv(h[3], h[4])
        self.conv5  = GCNConv(h[4], h[5])
        self.conv6  = GCNConv(h[5], h[6])
        self.conv7  = GCNConv(h[6], h[7])
        self.conv8  = GCNConv(h[7], h[8])
        self.conv9  = GCNConv(h[8], h[9])
        self.conv10 = GCNConv(h[9], h[10])
        self.conv11 = GCNConv(h[10], h[11])
        self.conv12 = GCNConv(h[11], h[12])

        self.linear1 = nn.Linear(h[12], h[13])
        self.linear2 = nn.Linear(h[13], h[14])
    
        self.bn1 = nn.BatchNorm1d(h[1])
        self.bn2 = nn.BatchNorm1d(h[2])
        self.bn3 = nn.BatchNorm1d(h[3])
        self.bn4 = nn.BatchNorm1d(h[4])
        self.bn5 = nn.BatchNorm1d(h[5])
        self.bn6 = nn.BatchNorm1d(h[6])
        self.bn7 = nn.BatchNorm1d(h[7])
        self.bn8 = nn.BatchNorm1d(h[8])
        self.bn9 = nn.BatchNorm1d(h[9])
        self.bn10 = nn.BatchNorm1d(h[10])
        self.bn11 = nn.BatchNorm1d(h[11])
        self.bn12 = nn.BatchNorm1d(h[12])

        self.l_relu = nn.LeakyReLU()
            

    def forward(self, data):

        z2, x_pos, edge_index = data.z2.to(self.device), data.x_pos.to(self.device), data.face_index.to(self.device)
        #n2 = torch.randn(z2.shape[0], z2.shape[1]).to(self.device) * 0.01
        dx = self.l_relu(self.bn1(self.conv1(z2, edge_index)))
        dx = self.l_relu(self.bn2(self.conv2(dx, edge_index)))
        dx = self.l_relu(self.bn3(self.conv3(dx, edge_index)))
        dx = self.l_relu(self.bn4(self.conv4(dx, edge_index)))
        dx = self.l_relu(self.bn5(self.conv5(dx, edge_index)))
        dx = self.l_relu(self.bn6(self.conv6(dx, edge_index)))
        dx = self.l_relu(self.bn7(self.conv7(dx, edge_index)))
        dx = self.l_relu(self.bn8(self.conv8(dx, edge_index)))
        dx = self.l_relu(self.bn9(self.conv9(dx, edge_index)))
        dx = self.l_relu(self.bn10(self.conv10(dx, edge_index)))
        dx = self.l_relu(self.bn11(self.conv11(dx, edge_index)))
        dx = self.l_relu(self.bn12(self.conv12(dx, edge_index)))
        
        dx = self.l_relu(self.linear1(dx))
        dx = torch.tanh(self.linear2(dx))
        
        dx_norm = torch.reciprocal(torch.norm(dx, dim=1, keepdim=True).expand(-1, 3) + 1.0e-12)
        x = torch.mul(dx, dx_norm)
        return x