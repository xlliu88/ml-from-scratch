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

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
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
        return self * (other ** -1)

    def __neg__(self):
        return self * Node(-1)

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self + (-other)
    
    def exp(self):
        out = Node(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
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
        #print(f'visited order {visited}')
        #print(f'topo list: {topo}')

        self.grad = 1.0
        for nod in reversed(topo):
            nod._backward()

    def __repr__(self):
        return(f'Node(data = {self.data}; grad = {self.grad})')





