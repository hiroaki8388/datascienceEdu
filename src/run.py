#%%
from toolz import pipe
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from src import layer, opt

%load_ext autoreload

%autoreload 2

#%%
def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # クラスごとのサンプル数
    DIM = 2  # データの要素数
    CLS_NUM = 3  # クラス数

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int)

    for j in range(CLS_NUM):
        for i in range(N):#N*j, N*(j+1)):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t
#%%
x, t =load_data()
print(x.shape)
print(t.shape)

#%%
plt.scatter(*x.T, c=t)

#%%
from src.layer import Affine, Sigmoid, SoftmaxWithLoss
class TwoLayerNet:
    def __init__(self, I, H, O):
        coef = 0.01
        W1 = coef*np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = coef*np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers =[
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = layer.SoftmaxWithLoss()

        self.params = sum([layer.params for layer in self.layers], [])
        self.grads = sum([layer.grads for layer in self.layers], [])

    def predict(self, x):
        funcs = [layer.forward for  layer in self.layers]

        return pipe(x, *funcs)
    
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t) 

        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        funcs = [layer.backward for layer in reversed(self.layers)]
        
        return pipe(dout, *funcs)



#%%

# hyper parameter
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate =1.

# data
x, t = load_data()
category = t.shape[1]

# model
from src.opt import SGD

model = TwoLayerNet(x.shape[1],hidden_size,category)
optimizer = SGD(learning_rate)
max_iters = len(x)// batch_size

total_loss = 0
loss_count = 0
loss_list = []


# 実行

for epoch in range(max_epoch):
    idx = np.random.permutation(len(x))
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters+1)%10 == 0:
            ave_loss = total_loss/loss_count
            print(f'epoch={epoch:d},iter={iters:d},ave_loss={ave_loss:.2f}')
            loss_list.append(ave_loss)
            total_loss, loss_count = 0,0
        



#%%
# 損失の可視化
plt.plot(loss_list)

#%%
# 予測の可視化
h = 0.01
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
pred_cls = np.argmax(score, axis=1)
Z = pred_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.5)
plt.scatter(*x.T, c=t)
plt.axis('off')

#%%
model.predict(x)
