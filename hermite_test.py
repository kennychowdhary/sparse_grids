from numpy import *
from numpy.polynomial import *
from matplotlib.pyplot import *
from scipy.misc import factorial
from hermite1d import *
import pdb

''' test weights for standard distribution '''
quad,weights = hermite_e.hermegauss(8)
norm_weights = weights/sqrt(2*pi)

mu = sum(quad * norm_weights)
sigma_sq = sum((quad-mu)**2 * norm_weights)
print mu, sigma_sq

# orthonormal Hermite polynomials w.r.t. st. normal
f0 = lambda x: 1 + 0*x
f1 = lambda x: x/sqrt(factorial(1))
f2 = lambda x: (x**2 - 1)/sqrt(factorial(2))
f3 = lambda x: (x**3 - 3*x)/sqrt(factorial(3))

print sum(f2(quad)**2 * norm_weights),
print sum(f3(quad)**2 * norm_weights)

''' test weights for shifted mean '''
quad,weights = hermite_e.hermegauss(8)
norm_weights = weights/sqrt(2*pi)

mu0 = 1.2
quad2 = quad + mu0
mu2 = sum(quad2 * norm_weights)
sigma_sq2 = sum((quad2-mu2)**2 * norm_weights)
print '\n', mu2, sigma_sq2

# orthonormal Hermite polynomials w.r.t. shifted mean
f0 = lambda x: 1 + 0*(x-mu0)
f1 = lambda x: (x-mu0)/sqrt(factorial(1))
f2 = lambda x: ((x-mu0)**2 - 1)/sqrt(factorial(2))
f3 = lambda x: ((x-mu0)**3 - 3*(x-mu0))/sqrt(factorial(3))

print sum(f2(quad2)*f3(quad2) * norm_weights),
print sum(f2(quad2)**2 * norm_weights),
print sum(f3(quad2)**2 * norm_weights)

# test weights for scaled st. dev only
mu0 = 0.0
sig0 = 2.0
quad3 = sig0*quad
mu3 = sum(quad3 * norm_weights)
sigma_sq3 = sum((quad3-mu3)**2 * norm_weights)
print '\n', mu3, sigma_sq3

# orthonormal Hermite polynomials w.r.t. shifted mean
f0 = lambda x: 1 + 0*(x/sig0)
f1 = lambda x: (x/sig0)/sqrt(factorial(1))
f2 = lambda x: ((x/sig0)**2 - 1)/sqrt(factorial(2))
f3 = lambda x: ((x/sig0)**3 - 3*(x/sig0))/sqrt(factorial(3))

print sum(f2(quad3)*f3(quad3) * norm_weights),
print sum(f2(quad3)**2 * norm_weights),
print sum(f3(quad3)**2 * norm_weights)

# test weights for scaled st. dev and shifted mean
mu0 = 1.0
sig0 = 2.0
quad4 = sig0*quad+mu0
mu4 = sum((sig0*quad+mu0) * norm_weights)
sigma_sq4 = sum(((sig0*quad+mu0)-mu4)**2 * norm_weights)
print '\n', mu4, sigma_sq4

# orthonormal Hermite polynomials w.r.t. shifted mean
f0 = lambda x: 1 + 0*((x-mu0)/sig0)
f1 = lambda x: ((x-mu0)/sig0)/sqrt(factorial(1))
f2 = lambda x: (((x-mu0)/sig0)**2 - 1)/sqrt(factorial(2))
f3 = lambda x: (((x-mu0)/sig0)**3 - 3*((x-mu0)/sig0))/sqrt(factorial(3))

print sum(f2(quad4)*f3(quad4) * norm_weights),
print sum(f2(quad4)**2 * norm_weights),
print sum(f3(quad4)**2 * norm_weights)

''' Test class '''
# approximate f(x)
mu = .25
sig = 1.5
test = hermite1d(mu,sig)
x = linspace(-5,5,100)
f = lambda x: sin((x-mu)/sig) + 3.5
nq = 16
qi,wi = test.quad(nquad=nq)
c = test.approx(fi=f(qi),deg=nq)
#c = test.approx(nquad=32,f=f,deg=32)
plot(x,f(x),x,test.eval(c,x))
print '\n', sum(f(qi)*wi), sqrt(sum((f(qi)-sum(f(qi)*wi))**2 * wi))
print c[0], sqrt(sum(c[1:]**2))
samples = f(sig*random.randn(1e5) + mu)
print mean(samples), sqrt(var(samples))




pdb.set_trace()