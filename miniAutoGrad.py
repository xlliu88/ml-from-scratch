'''
A miniature implementation of backpropagation auto-grade
only support scaler
'''

import numpy as np

class Node():
    def __init__(self, data, _inputs = ()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_inputs)
        self._backward = lambda:None

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        other = other.data if isinstance(other, Node) else Node(other)
        return self.__add__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)

    def __rsub__(self, other):
        return (self.__sub__(other))*Node(-1)
    
    def __neg__(self):
        return self * Node(-1)

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __pow__(self, other):
        # Ensure other is an int or float for simple power rule
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
         
        out = Node(self.data ** other, (self,))

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        assert other.data !=0, 'invalid operation: divide by 0'
        return self * (other ** -1)

    def __rtruediv__(self, other):
        assert self.data != 0, 'invalid operation: divide by 0'
        return other * self ** (-1)

    def exp(self):
        out = Node(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        '''
        default to natual log
        '''
        out = Node(np.log(self.data), (self,))

        def _backward():
            self.grad += 1/self.data * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Node(np.exp(self.data)/(np.exp(self.data) + 1), (self, ))

        def _backward():
            self.grad += out.data * (1-out.data) * out.grad
        out._backward = _backward
        return out

    def sin(self):
        '''
        x should be radian
        '''
        out = Node(np.sin(self.data), (self, ))

        def _backward():
            self.grad += np.cos(self.data) * out.grad
        out._backward = _backward
        return out
    
    def cos(self):
        '''
        x should be radian
        '''
        out = Node(np.cos(self.data), (self, ))

        def _backward():
            self.grad += -np.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def tan(self):
        '''
        x should be radian
        '''
        return self.sin()/self.cos()
     
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for cld in v._prev:
                    build_topo(cld)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for nod in reversed(topo):
            nod._backward()

    def __repr__(self):
        return(f'Node(data = {self.data}; grad = {self.grad})')

def log(x):
    if isinstance(x, Node):
        return x.log()
    else:
        return np.log(x)

def exp(x):
    if isinstance(x, Node):
        return x.exp()
    else:
        return np.exp(x)

def sigmoid(x):
    if isinstance(x, Node):
        return x.sigmoid()
    else:
        return 1/(1+np.exp(-x))

def sin(x):
    '''
    x should be radian
    '''
    if isinstance(x, Node):
        return x.sin()
    else:
        return np.sin(x)

def cos(x):
    '''
    x should be radian
    '''
    if isinstance(x, Node):
        return x.cos()
    else:
        return np.cos(x)

def tan(x):
    '''
    x should be radian
    '''
    if isinstance(x, Node):
        return x.tan()
    else:
        return np.tan(x)

if __name__ == "__main__":
    print('\033[31m-- auto grade mini-samples -------------------------------------\033[0m')
    print('\033[32mevaluate function: \033[32m \033[33m e^(e^x + e^(2y)) + sigmoid(e^x + e^(2y)) \033[0m at \033[33m (-1, 2)\033[0m:')
    x = Node(-1)
    y = Node(2)
    a = exp(x)
    b = exp(2*y)
    c = a + b
    d = exp(c)
    e = c.sigmoid()
    f = d + e

    f.backward()
    print('Node f:\t', f) 
    print('Node e:\t', e) 
    print('Node d:\t', d) 
    print('Node c:\t', c) 
    print('Node b:\t', b) 
    print('Node a:\t', a) 
    print('Node y:\t', y)
    print('Node x:\t', x)
    print()

    print('\033[32mevaluate sigmoid(x) at x = 2:\033[0m')
    x = Node(2)
    a = exp(-x)
    f = 1/(1 + a)
    f.backward()
    print('sigmoid x:\t', f)
    print('        x:\t', x)
    print('s(x)(1-s(x)):  \033[31m[should be the same as local gradient of x]\033[0m\n\t\t', \
          sigmoid(x)*(1-sigmoid(x)))

    print()
    
    print('\033[32mevaluate function: \033[32m \033[33m sin(e^x) + cos(e^y) + tan(e^z) \033[0m at degrees \033[33m (60, 30, 45)\033[0m:')
    x = Node(np.radians(60))
    y = Node(np.radians(30))
    z = Node(np.radians(45))

    a = exp(x)
    b = sin(a)
    c = exp(y)
    d = cos(c)
    e = exp(z)
    f = tan(e)
    g = a*b + c*d + e*f

    g.backward()

    print('Node g:\t', g) 
    print('Node f:\t', f) 
    print('Node e:\t', e) 
    print('Node d:\t', d) 
    print('Node c:\t', c) 
    print('Node b:\t', b) 
    print('Node a:\t', a) 
    print('Node y:\t', y)
    print('Node x:\t', x)
