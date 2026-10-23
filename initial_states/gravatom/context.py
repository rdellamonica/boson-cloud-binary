"""Symbolic parameters with numerical values (mu, M, alpha, ...) shared by the atomic states."""

import sympy as sp
import operator

class ValueSymbol:
    def __init__(self, symb, value, real = True, positive = True):
        self.symb = sp.symbols(symb, real = real, positive = positive)
        self.value = float(value)
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        self._value = value
    
    def __float__(self):
        return self.value
    
    def __repr__(self):
        return f'<{str(self.symb)} = {self.value}>'
    
    def __str__(self):
        return str(self.symb)
    
    def __add__(self, other):
        return DerivedSymbol([self, other], operator = operator.add)
    
    def __radd__(self, other):
        return DerivedSymbol([self, other], operator = operator.add)
    
    def __sub__(self, other):
        return DerivedSymbol([self, other], operator = operator.sub)
    
    def __mul__(self, other):
        return DerivedSymbol([self, other], operator = operator.mul)
    
    def __rmul__(self, other):
        return DerivedSymbol([self, other], operator = operator.mul)
    
    def __truediv__(self, other):
        return DerivedSymbol([self, other], operator = operator.truediv)
    
    def __pow__(self, other):
        return DerivedSymbol([self, other], operator = operator.pow)
    
class DerivedSymbol(ValueSymbol):
    
    operator_symbols_lookup = {
        operator.add: '+',
        operator.truediv: '/',
        operator.sub: '-',
        operator.mul: '*',
        operator.pow: '^',
    }
    
    def __init__(self, primitives, operator, symbol = None, real = True, positive = True):
        
        self.primitives = primitives
        self.operator = operator
        
        if symbol is None:
            self.symb = sp.symbols(self.representation)
        else:
            self.symb = sp.symbols(symbol, real = real, positive = positive)
    
    @property
    def symb(self):
        return self._symb
    
    @property
    def representation(self):
        return str(self.primitives[0])+self.operator_symbols_lookup.get(self.operator)+str(self.primitives[1])
    
    def __repr__(self):
        return f'<{self.symb}({self.representation}) = {self.value}>'
    
    @symb.setter
    def symb(self, value):
        if type(value) == str:
            self._symb = sp.symbols(value)
        elif type(value) == sp.Symbol:
            self._symb = value
        else:
            raise ValueError("Symbols must be either a string or a sympy.Symbol.")
    
    @property
    def value(self):
        return self.operator(*[float(primitive) for primitive in self.primitives])
    
    @value.setter
    def value(self):
        raise ValueError("Cannot change value of DerivedSymbol, change the primitive symbol instead")
    
    def __float__(self):
        return self.value
        
class Context:
    instances = []
    active = None
    
    def __init__(self, **kwargs):
        self.instances.append(self)
        self.set_active(self)
        self.context_dict = {}
        
        for (key, val) in zip(kwargs.keys(), kwargs.values()):
            val_symb = ValueSymbol(key, val)
            setattr(self, key, val_symb)
            self.context_dict[key] = val_symb
    
    def __call__(self, **kwargs):
        for (key, val) in zip(kwargs.keys(), kwargs.values()):
            if key in self.context_dict:
                if isinstance(val, DerivedSymbol):
                    val.symb = key
                    setattr(self, key, val)
                    self.context_dict[key] = val
                else:
                    self.context_dict[key].value = float(val)
            else:
                if isinstance(val, DerivedSymbol):
                    val.symb = key
                    setattr(self, key, val)
                    self.context_dict[key] = val
                    
                else:
                    val_symb = ValueSymbol(key, val)
                    setattr(self, key, val_symb)
                    self.context_dict[key] = val_symb

    def get_subs(self, excluded = []):
        return [(k.symb, k.value) for k in self.context_dict.values() if k.symb not in excluded]
    
    @property
    def symbols(self):
        return [k.symb for k in self.context_dict.values()]
    
    @property
    def values(self):
        return [k.value for k in self.context_dict.values()]

    @classmethod
    def get_active(cls):
        return cls.active

    @classmethod
    def set_active(cls, instance):
        cls.active = instance