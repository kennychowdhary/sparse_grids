import numpy.polynomial as poly
import matplotlib.pyplot as pl
from scipy.misc import factorial
import numpy as np

class hermite1d():
	'''
	class for probabilists orthonormal hermite polynomial
	'''
	def __init__(self,mu=0.0,sigma=1.0):
		self.mean = mu
		self.sigma = sigma
	def quad(self,nquad):
		'''
		Computes quadrature points and weights. 
		Can be used to integrate 2*nquad - 1 polynomials exactly
		'''
		q0,w0 = poly.hermite_e.hermegauss(nquad)
		wi = w0/np.sqrt(2*np.pi)
		qi = self.sigma*q0 + self.mean
		return qi,wi
	def eval(self,c,x):
		'''
		Evaluate hermite series with coefficient matrix x
		Normalization incorporated into c
		'''
		norm = np.array([np.sqrt(factorial(n)) for n in range(len(c))])
		return poly.hermite_e.hermeval((x-self.mean)/self.sigma,c/norm)
	def Hn(self,n,x):
		'''
		Evaluate nth hermite orthonormal polynomial
		'''
		c = np.zeros(n+1); c[-1:]=1
		return self.eval(c,x)
	def plotHn(self,n,xr=[-3,3],npnts=100):
		'''
		Plot nth degree orthonormal hermite polynomial
		'''
		x = np.linspace(xr[0]*self.sigma+self.mean,xr[1]*self.sigma+self.mean,npnts)
		y = self.Hn(n,x)
		pl.plot(x,y,'.')
		return x,y
	def approx(self,nquad=None,f=None,fi=None,deg=None):
		'''
		Approximates function f(x) with a nquad points of hermite polynomial
		Recall that with nquad points, we can approximate 2*nquad - 1 polynomial exactly
		'''
		if deg == None: deg = nquad
		c = np.zeros(deg+1)
		if fi == None:
			qi,wi = self.quad(nquad)
			fi = f(qi)
		else:
			nquad = len(fi)
			qi,wi = self.quad(nquad)
		for i in range(deg+1): c[i] = np.sum(fi*self.Hn(i,qi) * wi)
		return c









