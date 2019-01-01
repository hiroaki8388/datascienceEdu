import numpy as np
from toolz import pipe




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
        dx = dout*(1.-self.out_)*self.out_
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
        dW = self.x_.T@dout
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
        self.x_ = x

        return x@W

    def backward(self, dout):
        W, = self.params

        dx = dout@W.T 
        dW = x.T@dout

        # 深いコピー(初期化の際に確保した領域に上書き)
        self.grads[0][...] = dW

        return dx
        
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
