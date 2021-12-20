import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.norm = nn.LayerNorm(out_feats)
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x, adj):
        x = x.matmul(self.weight)
        x = adj.matmul(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class GraphModule(nn.Module):
    def __init__(self, num_layers, num_feats):
        super().__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

        layers = []
        for i in range(num_layers):
            layers.append(GCNLayer(num_feats, num_feats))
        self.gcn = nn.ModuleList(layers)

    def forward(self, x, device, get_adj=False):
        qx = self.wq(x)
        kx = self.wk(x)
        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)

        for layer in self.gcn:
            x = layer(x, adj)

        x = x.mean(dim=-2)
        if get_adj is False:
            return x
        else:
            return x, adj


class ClassifierSimple(nn.Module):
    def __init__(self, num_feats, num_hid, num_class):
        super().__init__()
        self.fc1 = nn.Linear(num_feats, num_hid)
        self.fc2 = nn.Linear(num_hid, num_class)
        self.drop = nn.Dropout()

    def forward(self, x, device):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ModelGCNConcAfter(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(2 * num_feats, num_feats, num_class)

    def forward(self, feats, feat_global, device, get_adj=False):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)

        if get_adj is False:
            x = self.graph(feats, device)
            x = x.view(N, FR, -1)
            x = self.graph(x, device)
            y = self.graph(feat_global, device)
            x = torch.cat([x, y], dim=-1)
            x = self.cls(x, device)
            return x
        else:
            x, adjobj = self.graph(feats, device, get_adj)
            adjobj = adjobj.cpu()
            wids_objects = adjobj.numpy().sum(axis=1)
            x = x.view(N, FR, -1)

            x, adjframelocal = self.graph(x, device, get_adj)
            adjframelocal = adjframelocal.cpu()
            wids_frame_local = adjframelocal.numpy().sum(axis=1)

            y, adjframeglobal = self.graph(feat_global, device, get_adj)
            adjframeglobal = adjframeglobal.cpu()
            wids_frame_global = adjframeglobal.numpy().sum(axis=1)

            x = torch.cat([x, y], dim=-1)
            x = self.cls(x, device)

            return x, wids_objects, wids_frame_local, wids_frame_global


class ModelGCNConcAfterGlobalOnly(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats, feat_global, device):

        x = self.graph(feat_global, device)
        x = self.cls(x, device)

        return x


class ModelGCNConcAfterLocalOnly(nn.Module):
    def __init__(self, gcn_layers, num_feats, num_class):
        super().__init__()
        self.graph = GraphModule(gcn_layers, num_feats)
        self.cls = ClassifierSimple(num_feats, int(num_feats/2), num_class)

    def forward(self, feats, feat_global, device):
        N, FR, B, NF = feats.shape
        feats = feats.view(N * FR, B, NF)

        x = self.graph(feats, device)
        x = x.view(N, FR, -1)
        x = self.graph(x, device)
        x = self.cls(x, device)

        return x
