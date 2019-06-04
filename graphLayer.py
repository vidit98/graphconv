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
	def __init__(self, h=64, w=64, d=2048, V=32,outfeatures=256):
		super(GCU, self).__init__()
		
		
		self.ht = h
		self.wdth= w
		self.d = d
		self.no_of_vert = V
		self.outfeatures = outfeatures

		self.W = Parameter(torch.cuda.FloatTensor(d,V))
		self.W.requires_grad = True
		self.W.retain_grad()
		self.variance = Parameter(torch.cuda.FloatTensor(d,V))
		self.variance.requires_grad = True
		self.variance.retain_grad()
		self.weight = Parameter(torch.cuda.FloatTensor(d, outfeatures))
		self.weight.requires_grad = True
		torch.nn.init.xavier_uniform(self.weight)
		torch.nn.init.xavier_uniform(self.W)
		torch.nn.init.xavier_uniform(self.variance)

		
		self.count = 0
		
		
		

	def init_param(self,x):
		vari = x.clone()
		x.detach()
		c = np.reshape(vari.detach().numpy(), (self.ht*self.wdth, self.d))
		kmeans = KMeans(n_clusters=self.no_of_vert, random_state=0).fit(c)
		W = Parameter(torch.t(torch.tensor(np.array(kmeans.cluster_centers_).astype('float32'))))
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

		variance = Parameter(torch.tensor((sig2 - sig1 + 1e-6	).astype('float32')))

		return W, variance
		# print("Printing variance\n", self.variance)


	def GraphProject(self,X):

		Adj = torch.cuda.FloatTensor(self.no_of_vert,self.no_of_vert)
		Z = torch.cuda.FloatTensor(self.d,self.no_of_vert)
		
		Q = torch.cuda.FloatTensor(self.ht*self.wdth,self.no_of_vert)

		for i in range(self.no_of_vert):
			q1 = self.W[:,i]
			# print(q1)
			q = X - q1[None,None,:]
			# print("after subracting", q)
			# print("variance", self.variance[:,i])
			q = q/self.variance[:,i]
			
			q = torch.reshape(q,(self.ht*self.wdth, self.d))

			q = q**2
			# print("after dividing", q)
			q = torch.sum(q, dim=1)
			# print("Prinitng\n", q)
			# q = torch.exp(-q*0.5)
			Q[:,i] = q
			# print("Prinitng q\n", self.Q[:,i])
		# print(torch.max(self.Q, 1)[0])
		Q -= torch.min(Q, 1)[0][:,None]
		Q = torch.exp(-Q*0.5)
		norm = torch.sum(Q, dim=1)
		# print(norm.shape)
		Q = torch.div(Q,norm[:,None])

		# print("Printing Q\n",self.Q)

		

		print("the pixel-to-vertex assignment matrix Done")
		for i in range(self.no_of_vert):
			z1 = self.W[:,i]
			q = X - z1[None,None,:]
			z = q/self.variance[:,i]
			z = torch.reshape(z,(self.ht*self.wdth, self.d))
			
			z = torch.mul(z,Q[:,i][:,None])

			z = torch.sum(z,dim=0)

			n = torch.sum(Q[:,i],dim=0)
			if torch.equal(z,torch.cuda.FloatTensor(z.shape).fill_(0)) and torch.equal(n,torch.cuda.FloatTensor(n.shape).fill_(0)):
				z = torch.ones(z.shape)
			else:
				z = z/n
			Z[:,i] = z

		norm = Z**2

		norm = torch.sum(norm, dim=0)
		# print("norm size",norm.shape)
		# print("Z \n",self.Z)
		Z = torch.div(Z,norm)

		#print("the vertex features Z Done")
		Adj = torch.mm(torch.t(Z), Z)
		#print("the adjacency matrix A Done")

		return (Q, Z, Adj)

	def GraphReproject(self, Z_o,Q):
		X_new = torch.mm(Z_o,Q)
		return torch.reshape(X_new, (self.ht, self.wdth, self.outfeatures))

	def forward(self, X):
		X = torch.reshape(X,(self.ht, self.wdth, self.d)).float()
		# if self.count == 0:
		# 	with torch.no_grad():
		# 		self.W, self.variance = self.init_param(X)
		# 	self.count += 1
		
		
		
		Q, Z, Adj = self.GraphProject(X)
		# print("Prinitng Z\n",self.Z)

		out = torch.mm(torch.t(Z), self.weight)
		out = torch.spmm(Adj, out)
		Z_o = F.relu(out)

		out = self.GraphReproject(Q, Z_o)
		out = out.view(1, self.outfeatures, self.ht, self.wdth) #usample requires 4Dtensor
		#print("Prinitng Z", Z.shape)
		
		return out







		
