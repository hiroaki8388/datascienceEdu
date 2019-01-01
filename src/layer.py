import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from toolz import pipe
sns.set()




class Layer:
    def __init__(self, params=[], grads=[]):
        self.params = params
        self.grads = grads
    
    def forward(self, x):
        pass
    
    def backward(self, dout):
        pass

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out_ = out

        return out 
    
    def backward(self, dout):
        dx = dout*(1.-self.out)*self.out_
        return dx

class Affine(Layer):
    def __init__(self, W, b):
        grads = [np.zeros_like(W), np.zeros_like(b)]
        super().__init__(params=[W, b], grads=grads)
    
    def forward(self, x):
        W, b = self.params
        out = x@W+b
        self.x_ = x

        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = dout@W.T
        dW = x.T@W 
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

    
class MatMul(Layer):
    def __init(self, W):
        grads = [np.zeros_like(W)] 
        super().__init__([W], grads)
    
    def forward(self, x):
        W, = self.params
        self.x_ = x_

        return x_@W

    def backward(self, dout):
        W, = self.params

        dx = dout@W.T 
        dW = x.T@W

        # 深いコピー(初期化の際に確保した領域に上書き)
        self.grads[0][...] = dW

        return dx
        

class TwoLayerNet:
    def __init__(self, I, H, O):
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers =[
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.params = sum([layer.params for layer in self.layers], [])

    def predict(self, x):
        funcs = [layer.forward for  layer in self.layers]

        return pipe(x, *funcs)
