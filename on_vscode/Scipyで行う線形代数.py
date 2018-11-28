# %%
from collections import defaultdict
import itertools as it
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
%matplotlib inline

#%%
# 隣接行列を定義
A = np.array(
[[0, 1, 1, 0, 0, 0],
[1, 0, 1, 0, 0, 0],
[1, 1, 0, 1, 0, 0],
[0, 0, 1, 0, 1, 1],
[0, 0, 0, 1, 0, 1],
[0, 0, 0, 1, 1, 0]], dtype=float)

#%%
# 隣接行列を可視化
g=nx.from_numpy_matrix(A)
layout=nx.spring_layout(g,pos=nx.circular_layout(g))
nx.draw(g,pos=layout,with_labels=True,node_color='red')

#%%
# グラフのラプラシアンを作成
# 対象のnodeから伸びているedgeの数
D=np.diag(np.sum(A,axis=0))
L=D-A
print(L)

#%%
# ラプラシアンは対称行列なので、必ず実数の固有値を持つ固有ベクトルが存在する
vals,Vecs=np.linalg.eigh(L)

vec0=Vecs[:,1]
print(vals[1])
print(L@vec0/vec0)

#%%
# フィードラーベクトル=二番目に小さい固有値に対応する固有ベクトル
# の値の符号により、グループ分けができる
f=Vec[:,np.argsort(vals)[1]]
plt.plot(f,marker='o',lw=0)


#%%
# 実際に色分けしてみる
colors=['orange' if eigv >0 else 'gray' for eigv in f]
nx.draw(g,pos=layout,with_labels=True,node_color=colors)

#%%
# 脳データのラプラシアン
from  os.path import join
dir_='./dataset/elegant-scipy/data'
Chem=np.load(join(dir_,'chem-network.npy'))
Gap=np.load(join(dir_,'gap-network.npy'))

neuron_ids=np.load(join(dir_,'neurons.npy'))
neuron_types=np.load(join(dir_,'neuron-types.npy'))

# 隣接行列を作成
A=Chem+Gap
C=(A+A.T)/2

# 次数行列を作成
degrees=np.sum(C,axis=0)
D=np.diag(degrees)

# ラプラシアンを作成
L=D-C
L