import numpy as np
import matplotlib.pyplot as pl
import pdb
from collections import defaultdict


class U():
	'''
	Class for holding information about Umk using the trapezoidal rule on [0,1]
	'''
	def __init__(self,k):
		'''
		Input
		k : determines U_k
		Output:
		xi : effective quad pnts for U_k
		wi : effective weights corresponding to xi (should always be positive)
		XW : dictionary with xi:wi as the key:value pairsa
		'''
		self.k = k
		self.mk = 2**k - 1
		if k == 0:
			self.xi = np.array([0.0])
			self.wi = np.array([0.0])
		else:
			self.xi = (np.arange(0,self.mk)+1.0)/(self.mk+1.0)
			wi = np.ones(len(self.xi)); 
			wi[[0,len(self.xi)-1]] = 3./2.
			if k == 1:
				wi = 0*wi + 1.0
			else:
				wi = wi/(self.mk+1.0)
			self.wi = wi
		# returns dictionary with xi for the key, and wi for the element
		self.XW = {self.xi[i]:self.wi[i] for i in range(len(self.xi))}

class L(U):
	'''
	Class for computing the difference quadrature Uk - U_{k-1}
	'''
	def __init__(self,k):
		'''
		Input:
		k : determines L_k
		Output: 
		xi : effective quad pnts for L_k
		wi : effective weights corresponding to xi (may be negative)
		XW : dictionary with xi:wi as the key:value pairsa
		'''
		self.k = k
		if k == 1:
			self.xi = U(1).xi
			self.wi = U(1).wi
		else:
			Uk = U(k) # U_k
			Ukm1 = U(k-1) # U_{k-1}
			uk = {Uk.xi[i]: Uk.wi[i] for i in range(len(Uk.xi))}
			ukm1 = {Ukm1.xi[i]: -Ukm1.wi[i] for i in range(len(Ukm1.xi))}
			temp = defaultdict(list)
			for d in [ukm1, uk]: 
			    for key, value in d.iteritems():
			        temp[key].append(value)
			self.xi = np.array(temp.keys())
			self.wi = np.array([np.sum(temp[item]) for item in temp.keys()])
		# returns dictionary with xi for the key, and wi for the element
		self.XW = {self.xi[i]:self.wi[i] for i in range(len(self.xi))}


''' Define 2d tensor products using Meshgrid function '''

def tensorprod(u1,u2):
	'''
	Returns tensor product of two quadrature rules. Input is either U or L
	Output are two meshgrids. The first is a meshgrid of the coordinate pairs, and the second is a meshgrid of the weights, respively.
	'''
	xi1 = u1.xi
	xi2 = u2.xi
	wi1 = u1.wi
	wi2 = u2.wi
	return np.meshgrid(xi1,xi2), np.meshgrid(wi1,wi2)

def tensorprod2(u1,u2):
	'''
	Returns dictionary where keys are the coordinates, and values are the weights
	u1 is the x coordinate quadrature and u2 is the y coordinate quadrature
	'''
	dict1 = u1.XW
	dict2 = u2.XW
	master_xi = []
	master_wi = []
	master_dict = defaultdict(list)
	for key1,value1 in dict1.iteritems():
		for key2,value2 in dict2.iteritems():
			coord = key1,key2
			weights = value1,value2
			master_dict[coord].append(weights)
			master_xi.append(np.array(coord))
			master_wi.append(np.array(weights))
	x = [[item[i] for item in master_xi] for i in range(2)]
	w = [np.prod(item) for item in master_wi]
	return x,w,master_dict

''' Test sparse grids '''


# Smolyak grid
k = range(1,10)
k1,k2 = np.meshgrid(k,k)
kk = k1+k2
'''dim = 2
level = 3
maxK = level + dim - 1.0
k1 = k1[(kk<=maxK)]
k2 = k2[(kk<=maxK)]
pl.figure()
all_dict = []
for i in range(len(k1)):
	x,w,xw = tensorprod2(L(k1[i]),L(k2[i]))
	all_dict.append(xw)
	pl.plot(x[0],x[1],'o')
'''

# Tensor product grid
pl.figure()
all_dict = []
for i in range(1,5):
	for j in range(1,5):
		x,w,xw = tensorprod2(L(i),L(j))
		pl.plot(x[0],x[1],'o')
		all_dict.append(xw)



# combine dictionaries into one single master list
master_dict = defaultdict(list)
for element in all_dict:
	for key,value in element.iteritems():
		master_dict[key].append(value)
# combine weights is there are multiple weights
for key,value in master_dict.iteritems():
	if len(value) >= 1:
		master_dict[key] = [sum([item[0][i] for item in value]) for i in range(2)]

# test integration rule on [0,1]
f = lambda x,y: sin(2*np.pi*x)


pdb.set_trace()