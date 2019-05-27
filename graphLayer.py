import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module

#TODO
"""
Currently only for 1 iamge write a code to generalize for a batch
"""

class GCU(Module):
	def __init__(self, X=None, V=32,outfeatures=256):
		super(GCU, self).__init__()
		self.X = X
		_,d,h, w = X.shape
		self.ht = h
		self.wdth= w
		self.d = d
		self.no_of_vert = V
		self.outfeatures = outfeatures

		self.W = Parameter(torch.randn(d,V))
		self.W.requires_grad = True
		self.variance = Parameter(torch.randn(d,V))
		self.variance.requires_grad = True
		self.weight = Parameter(torch.FloatTensor(d, outfeatures))
		self.weight.requires_grad = True

		self.Adj = torch.FloatTensor(V,V)
		self.Z = torch.FloatTensor(d,V)
		self.Z_o = None
		self.Q = torch.FloatTensor(h*w,V)
		self.count = 0
		
		# self.conv1 = torch.nn.Conv2d(outfeatures, 3, kernel_size=3, stride=1, padding=1)

		

	def init_param(self):

		c = np.reshape(self.X.numpy(), (self.ht*self.wdth, self.d))
		kmeans = KMeans(n_clusters=self.no_of_vert, random_state=0).fit(c)
		self.W = Parameter(torch.t(torch.tensor(np.array(kmeans.cluster_centers_).astype('float32'))))
		# print("Printing W\n", self.W)
		lab = kmeans.labels_
		c_s = np.square(c)
		sig1 = np.zeros((self.d, self.no_of_vert))
		sig2 = np.zeros((self.d, self.no_of_vert))
		count = np.array([0 for i in range(self.no_of_vert)])
		for i in range(len(lab)):
			sig1[:,lab[i]] += np.transpose(c[i])
			sig2[:,lab[i]] += np.transpose(c_s[i])
			count[lab[i]] += 1

		sig2 = sig2/count
		sig1 = sig1/count

		sig1 = np.square(sig1)

		self.variance = Parameter(torch.tensor((sig2 - sig1 + 1e-6	).astype('float32')))
		# print("Printing variance\n", self.variance)


	def GraphProject(self):

		for i in range(self.no_of_vert):
			q1 = self.W[:,i]
			# print(q1)
			q = self.X - q1[None,None,:]
			# print("after subracting", q)
			# print("variance", self.variance[:,i])
			q = q/self.variance[:,i]
			
			q = torch.reshape(q,(self.ht*self.wdth, self.d))

			q = q**2
			# print("after dividing", q)
			q = torch.sum(q, dim=1)
			# print("Prinitng\n", q)
			# q = torch.exp(-q*0.5)
			self.Q[:,i] = q
			# print("Prinitng q\n", self.Q[:,i])
		# print(torch.max(self.Q, 1)[0])
		self.Q -= torch.min(self.Q, 1)[0][:,None]
		self.Q = torch.exp(-self.Q*0.5)
		norm = torch.sum(self.Q, dim=1)
		# print(norm.shape)
		self.Q = torch.div(self.Q,norm[:,None])
		# print("Printing Q\n",self.Q)

		

		print("the pixel-to-vertex assignment matrix Done")
		for i in range(self.no_of_vert):
			z1 = self.W[:,i]
			q = self.X - z1[None,None,:]
			z = q/self.variance[:,i]
			z = torch.reshape(z,(self.ht*self.wdth, self.d))
			
			z = torch.mul(z,self.Q[:,i][:,None])

			z = torch.sum(z,dim=0)

			n = torch.sum(self.Q[:,i],dim=0)
			if torch.equal(z,torch.zeros(z.shape)) and torch.equal(n,torch.zeros(n.shape)):
				z = torch.ones(z.shape)
			else:
				z = z/n
			#print(self.Z[:,i].shape, self.Z.shape, self.ht, self.wdth, self.d)
			self.Z[:,i] = z

		norm = self.Z**2

		norm = torch.sum(norm, dim=0)
		# print("norm size",norm.shape)
		# print("Z \n",self.Z)
		self.Z = torch.div(self.Z,norm)

		print("the vertex features Z Done")
		self.Adj = torch.mm(torch.t(self.Z), self.Z)
		print("the adjacency matrix A Done")

	def GraphReproject(self):
		X_new = torch.mm(self.Q, self.Z_o)
		return torch.reshape(X_new, (self.ht, self.wdth, self.outfeatures))

	def forward(self, X):
		if self.count == 0:
			self.init_param()
			self.count += 1
		self.X = torch.reshape(X,(self.ht, self.wdth, self.d)).float()
		print("Printing X", self.X.shape)
		print("Prinitng Z", self.Z.shape)
		self.GraphProject()
		# print("Prinitng Z\n",self.Z)

		out = torch.mm(torch.t(self.Z), self.weight)
		out = torch.spmm(self.Adj, out)
		self.Z_o = F.relu(out)

		out = self.GraphReproject()
		out = out.view(1, self.outfeatures, self.ht, self.wdth) #usample requires 4Dtensor
		print("Prinitng Z", self.Z.shape)
		
		return out





"""

TODO convert numpy to tensor or vice versa 
import numpy as np
import torch
import torch.nn as nn
from model import ModelBuilder, SegmentationModule



"""




		
