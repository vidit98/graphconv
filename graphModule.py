import torch.nn as nn
from graphLayer import GCU
import torch


class GraphConv(nn.Module):
    def __init__(self, outfeatures=256):
        super(GraphConv, self).__init__()

        self.gc1 = GCU(V=2)
        self.gc2 = GCU(V=4)
        #self.gc3 = GCU(V=8)
        #self.gc4 = GCU(V=32)
      
    def forward(self, x):

        out = torch.cat((x,self.gc1(x)),dim=1)
        out = torch.cat((out,self.gc2(x)),dim=1)
        #out = torch.cat((out,self.gc3(x)),dim=1)

        #out = torch.cat((out,self.gc4(x)),dim=1)
        #out = self.gc1(x)
        return out