"""Hydrogenic bound and unbound states of a gravitational atom, as SymPy expressions."""

from .context import Context

import sympy as sp
import numpy as np
import os

import pickle

from sympy.functions.special.spherical_harmonics import Ynm
from sympy.functions.special.polynomials import assoc_laguerre
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import Heaviside

from scipy.integrate import quad
from scipy.interpolate import interp1d
from mpmath import hyp1f1, hyper

import warnings

module_dir = os.path.dirname(__file__)

def hyper1f1(a, b, z):
    return hyp1f1(a[0], b[0], z)

class Int_function:
    def __init__(self, name: str, n: int, l:int, m:int):
        
        self.x, self.F_co, self.F_counter = np.genfromtxt(module_dir + f'/functions/{name}_{n}{l}{m}.csv', delimiter = ',').T
        self.F_int_co = interp1d(self.x, self.F_co, bounds_error=False, fill_value=0)
        self.F_int_counter = interp1d(self.x, self.F_counter, bounds_error=False, fill_value=0)
    
    def __call__(self, x, counter = False):
        if not counter:
            return self.F_int_co(x)
        else:
            return self.F_int_counter(x)
    
class BoundState:
    
    def __init__(self, n: int, l: int, m: int, context: Context):
        
        self.n = n
        self.m = m
        self.l = l
        self.context = context
        
        self.t, self.r, self.theta, self.phi = sp.symbols('t r theta phi', real = True, positive = True)
        
        self.omega = sp.symbols(fr'\omega_{{{n}{l}{m}}}', real = True)
        
        if (l != 0) and (hasattr(context, 'a')):
            self.eigenvalue = context.mu.symb*(
                1 - 
                context.alpha.symb**2/(2*n**2)
                #context.alpha.symb**4/(8*n**4)
                #(3*n-2*l-1)*context.alpha.symb**4/(n**4*(l+1/2))#+
                #2*context.a.symb*m*context.alpha.symb**5/(n**3*l*(l+1/2)*(l+1)*context.M.symb)
            )
            
        else:
            self.eigenvalue = context.mu.symb*(
                1 - 
                context.alpha.symb**2/(2*n**2)
                #context.alpha.symb**4/(8*n**4)
                #(3*n-2*l-1)*context.alpha.symb**4/(n**4*(l+1/2))
            )
        
        self.omega_R = self.eigenvalue
        
        if hasattr(context, 'a'):
            self.r_plus = context.M.symb+sp.sqrt(context.M.symb**2-context.a.symb**2)
            self.Omega_H = context.a.symb/(2*context.M.symb*self.r_plus)
            self.omega_I = 2*self.r_plus/context.M.symb*self.C_nlm()*(m*self.Omega_H-self.eigenvalue)*context.alpha.symb**(4*l+5)
        
        radial = sp.sqrt((2*context.mu.symb*context.alpha.symb/n)**3*sp.factorial(n-l-1)/(2*n*sp.factorial(n+l)))*(2*context.mu.symb*context.alpha.symb/n*self.r)**l*sp.exp(-context.mu.symb*context.alpha.symb/n*self.r)*assoc_laguerre(n-l-1, 2*l+1, 2*context.mu.symb*context.alpha.symb/n*self.r)
        
        self.radial = radial
        
        self.eigenstate = radial*Ynm(l, m, self.theta, self.phi)*sp.exp(-sp.I*(self.omega-context.mu.symb)*self.t)
        
        eigenstate_subs = self.eigenstate.expand(func = True).subs([(self.omega, self.eigenvalue)])
        self.eigenstate_f = sp.lambdify([self.t, self.r, self.theta, self.phi, *context.symbols], eigenstate_subs)
        
        if os.path.exists(module_dir + f'/functions/P_{n}{l}{m}.csv'):
            self.P = Int_function('P', self.n, self.l, self.m)
            
        if os.path.exists(module_dir + f'/functions/R_{n}{l}{m}.csv'):
            self.R = Int_function('R', self.n, self.l, self.m)
        
    def __call__(self, t, r, theta, phi):
        return self.eigenstate_f(t, r, theta, phi, *self.context.values)

    def C_nlm(self):
        n = self.n
        l = self.l
        m = self.m
        
        if l != 0:
            if l > 1:
                j_s = np.arange(1, l+1)
            else:
                j_s = [1]
                
            K = 1
            
            for j in j_s:
                #K *= j**2*(1-(self.context.a.symb/self.context.M.symb)**2)+(self.context.a.symb/self.context.M.symb*m-2*self.r_plus/self.context.M.symb*self.context.alpha.symb)**2
                K *= j**2*(1-(self.context.a.symb/self.context.M.symb)**2)+4*self.r_plus**2*(m*self.Omega_H-self.eigenvalue)**2
        else:
            j = 1
            K = j**2*(1-(self.context.a.symb/self.context.M.symb)**2)+4*self.r_plus**2*(m*self.Omega_H-self.eigenvalue)**2
        
        return 2**(4*l+1)*sp.factorial(n+l)/(n**(2*l+4)*sp.factorial(n-l-1))*(sp.factorial(l)/(sp.factorial(2*l)*sp.factorial(2*l+1)))**2*K
        
    def rho(self, r, theta, phi):
        return abs(self(0, r, theta, phi))**2
    

class UnboundState:
    
    def __init__(self, k: float, l: int, m: int):
        
        context = Context.get_active()
        
        self.k_v = k
        
        k = sp.symbols('k', real = True, positive = True)
        
        self.k = k
        self.m = m
        self.l = l
        self.context = context
        
        self.t, self.r, self.theta, self.phi = sp.symbols('t r theta phi', real = True, positive = True)
        
        self.eigenvalue = k**2/(2*context.mu.symb)#sp.sqrt(context.mu.symb**2+k**2)-context.mu.symb#k**2/(2*context.mu.symb)
        
        self.R_kl_int = 2*k*sp.exp(sp.pi*context.mu.symb*context.alpha.symb/(2*k))*sp.Abs(sp.gamma(l+1+sp.I*context.mu.symb*context.alpha.symb/k))/sp.factorial(2*l+1)*(2*k*self.r)**l*sp.exp(-sp.I*k*self.r)*sp.hyper([l+1+sp.I*context.mu.symb*context.alpha.symb/k], [2*l+2], 2*sp.I*k*self.r)
        
        R_kl = sp.sqrt(4*sp.pi*k/self.r)*besselj(2*l+1, 2*sp.sqrt(2*context.alpha.symb*context.mu.symb*self.r))
        
        self.radial = R_kl
        
        self.eigenstate = R_kl*Ynm(l, m, self.theta, self.phi)*sp.exp(-sp.I*self.eigenvalue*self.t)
        
        eigenstate_subs = self.eigenstate.expand(func = True)
        self.eigenstate_f = sp.lambdify([self.t, self.r, self.theta, self.phi, self.k, *context.symbols], eigenstate_subs)
    
    def __call__(self, t, r, theta, phi):
        eigenstate_subs = self.eigenstate.expand(func = True).subs([(self.omega, self.eigenvalue)] + self.context.get_subs())
        eigenstate_f = sp.lambdify([self.t, self.r, self.theta, self.phi], eigenstate_subs)
        
        return eigenstate_f(t, r, theta, phi)
    
    def rho(self, r, theta, phi):
        return self.context.M_c.value*abs(self(0, r, theta, phi))