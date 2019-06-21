import torch.nn as nn
from graphLayer import GCU
import torch


class GraphConv(nn.Module):
    def __init__(self, batch, outfeatures=256):
        super(GraphConv, self).__init__()

        #self.gc1 = GCU(V=2, batch=batch)
        #self.gc2 = GCU(V=4, batch=batch)
        self.gc3 = GCU(V=2, batch=batch)
        #self.gc4 = GCU(V=32)
      
    def forward(self, x):
        y = self.gc3(x)
       # print(x.shape, y.shape)
        out = torch.cat((x,y),dim=1)
        #out = torch.cat((out,self.gc2(x)),dim=1)
        #out = torch.cat((out,self.gc3(x)),dim=1)

        #out = torch.cat((out,self.gc4(x)),dim=1)
        #out = self.gc1(x)
        return out