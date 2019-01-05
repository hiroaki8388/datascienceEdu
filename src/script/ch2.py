#%%
import sys
from src.util import preprocess, convert_one_hot
import numpy as np

from src.layer import MatMul, SoftmaxWithLoss

%matplotlib inline
%load_ext autoreload
%autoreload 2

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        eps = 0.01
        rnd = np.random
        W_in = eps*rnd.randn(V, H).astype('f')
        W_out = eps*rnd.randn(H, V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer, self.loss_layer]
    
        self.params = sum([layer.params for layer in layers],[])
        self.grads = sum([layer.grads for layer in layers],[])

        self.word_vecs = W_in

    def forward(self, context, target):
        h0 =self.in_layer0.forward(context[:, 0])
        h1 =self.in_layer1.forward(context[:, 1])
        h =(h0+h1)/2.
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)

        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)

        return None
#%%
txt = 'you say goodby and I say hello.'

corpus, word_to_id, id_to_word = preprocess(txt)
print(corpus)
print(word_to_id)
print(id_to_word)

#%%
def create_contexts_target(corpus, win_size=1):
    t = np.array(corpus[win_size:-win_size])

    head = corpus[:-(win_size+1)]
    tail = corpus[(win_size+1):]

    contexts = np.array([head, tail]).T

    return contexts, t

#%%
# 前処理
# contextとtに文章を変換
contexts, t = create_contexts_target(corpus)
vocab_size = len(corpus)

# contextとtをone-hot-encoding
contexts = convert_one_hot(contexts, vocab_size)
t = convert_one_hot(t, vocab_size)


#%%
# 学習を実行
from src.opt import Adam
from src.trainer import Trainer 
hidden_size =5
batch_size = 3
max_epoch = 1000

model = SimpleCBOW(vocab_size, hidden_size)
opt = Adam()
trainer = Trainer(model, opt)
trainer.fit(contexts,t, max_epoch=max_epoch, batch_size=batch_size)
trainer.plot()

#%%
# 分散表現
import matplotlib.pyplot as plt
ax = plt.imshow(model.word_vecs)








