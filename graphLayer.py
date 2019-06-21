import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module

#TODO
"""
Curre
"""

class GCU(Module):
	def __init__(self, batch = 16, h=64, w=64, d=2048, V=32,outfeatures=256):
		super(GCU, self).__init__()
		
		
		self.ht = h
		self.wdth= w
		self.d = d
		self.no_of_vert = V
		self.outfeatures = outfeatures
		self.batch = batch
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


		self.iden = torch.eye(self.d).cuda(1)
		self.iden = torch.cat((self.iden, self.iden))
		for i in range(int(np.log2(V))):
			self.iden = torch.cat((self.iden,self.iden), dim=1)

		
		
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

		Adj = torch.cuda.FloatTensor(self.batch, self.no_of_vert,self.no_of_vert)
		Adj = Adj.cuda(1)
		Z = torch.cuda.FloatTensor(self.batch,self.d,self.no_of_vert)
		Z = Z.cuda(1)
		Q = torch.cuda.FloatTensor(self.batch, self.ht*self.wdth,self.no_of_vert)

		Q = Q.cuda(1)
		#print("Hello",Z.get_device(), Q.get_device())
		for i in range(self.no_of_vert):
			q1 = self.W[:,i]
			# print(q1)
			q = X - q1[None,None,None,:]
			# print("after subracting", q)
			# print("variance", self.variance[:,i])
			q = q/self.variance[:,i]
			
			q = torch.reshape(q,(self.batch, self.ht*self.wdth, self.d))

			q = q**2
			# print("after dividing", q)
			q = torch.sum(q, dim=2)
			# print("Prinitng\n", q)
			# q = torch.exp(-q*0.5)
			Q[:,:,i] = q
			# print("Prinitng q\n", self.Q[:,i])
		# print(torch.max(self.Q, 1)[0])
		Q -= torch.min(Q, 2)[0][:,:,None]
		Q = torch.exp(-Q*0.5)
		norm = torch.sum(Q, dim=2)
		# print(norm.shape)
		Q = torch.div(Q,norm[:,:,None])

		# print("Printing Q\n",self.Q)

		

		#print("the pixel-to-vertex assignment matrix Done")
		for i in range(self.no_of_vert):
			z1 = self.W[:,i]
			q = X - z1[None,None,None,:]
			z = q/self.variance[:,i]
			z = torch.reshape(z,(self.batch, self.ht*self.wdth, self.d))
			
			z = torch.mul(z,Q[:,:,i][:,:,None])

			z = torch.sum(z,dim=1)

			n = torch.sum(Q[:,:,i],dim=1)
			#print(z.shape, n.shape)
			if torch.equal(z,torch.cuda.FloatTensor(z.shape).fill_(0).cuda(1)) and torch.equal(n,torch.cuda.FloatTensor(n.shape).fill_(0).cuda(1)):
				z = torch.ones(z.shape)
			else:
				z = z/n[:,None]
			Z[:,:,i] = z

		norm = Z**2

		norm = torch.sum(norm, dim=1)
		# print("norm size",norm.shape)
		# print("Z \n",self.Z)
		Z = torch.div(Z,norm[:,None])

		#print("the vertex features Z Done", torch.transpose(Z,1,2).shape, Z.shape)
		#torch.cuda.synchronize()
		Adj = torch.matmul(torch.transpose(Z,1,2), Z)
		#print("the adjacency matrix A Done")

		return (Q, Z, Adj)

	def GraphProject_optim(self, X):

		Adj = torch.cuda.FloatTensor(self.batch, self.no_of_vert,self.no_of_vert)
		Adj = Adj.cuda(1)
		Z = torch.cuda.FloatTensor(self.batch,self.d,self.no_of_vert)
		Z = Z.cuda(1)
		Q = torch.cuda.FloatTensor(self.batch, self.ht*self.wdth,self.no_of_vert)

		Q = Q.cuda(1)
		X = torch.reshape(X,(self.batch, self.ht*self.wdth, self.d))
		zero = torch.cuda.FloatTensor(X.shape).fill_(0).cuda(1) 
		new = torch.cat((zero, X), dim=2)
		#print("Shapes", new.shape, zero.shape, X.shape)
		extend = torch.matmul(new, self.iden)
		#print(extend.shape)

		W = torch.reshape(self.W, (self.d*self.no_of_vert,))
		q = extend - W[None,None,:]
		variance = torch.reshape(self.variance, (self.d*self.no_of_vert,))
		q1 = q/variance[None, None, :]
		q = q1**2
		q = torch.reshape(q, (self.batch, self.ht*self.wdth, self.d , self.no_of_vert))
		q = torch.sum(q, dim=2)
		q = torch.reshape(q, (self.batch, self.ht*self.wdth, self.no_of_vert))
		Q = q

		Q -= torch.min(Q, 2)[0][:,:,None]
		Q = torch.exp(-Q*0.5)
		norm = torch.sum(Q, dim=2)
		# print(norm.shape)
		Q = torch.div(Q,norm[:,:,None])

		#print(Q.shape)

		z = torch.reshape(q1, (self.batch,  self.d , self.ht*self.wdth , self.no_of_vert))
		z = torch.mul(z,Q)
		z = torch.sum(z, dim=2)
		#print("MIN",torch.min(torch.abs(z)), z.shape)
		z = torch.add(z, 10e-8)/torch.add(torch.sum(Q,dim=1), 10e-8)

		norm = z**2
		norm = torch.sum(norm, dim=1)
		# print("norm size",norm.shape)
		# print("Z \n",self.Z)
		Z = torch.div(Z,norm)

		#print("the vertex features Z Done")
		Adj = torch.matmul(torch.transpose(Z,1,2), Z)

		return (Q, Z, Adj)



	def GraphReproject(self, Z_o,Q):
		X_new = torch.matmul(Z_o,Q)
		return torch.reshape(X_new, (self.batch, self.outfeatures, self.ht, self.wdth))

	def forward(self, X):
		X = torch.reshape(X,(self.batch, self.ht, self.wdth, self.d)).float()
		# if self.count == 0:
		# 	with torch.no_grad():
		# 		self.W, self.variance = self.init_param(X)
		# 	self.count += 1
		
		
		
		Q, Z, Adj = self.GraphProject_optim(X)
		# print("Prinitng Z\n",self.Z)
		#print("Hello1", self.weight.get_device())
		out = torch.matmul(torch.transpose(Z,1,2), self.weight)
		out = torch.matmul(Adj, out)
		Z_o = F.relu(out)

		out = self.GraphReproject(Q, Z_o)
		#out = out.view(self.batch, self.outfeatures, self.ht, self.wdth) #usample requires 4Dtensor
		#print("Prinitng Z", Z.shape)
		
		return out







		
