#%%
import numpy as np
import pandas as pd
import scipy as sp
import toolz as tz
from toolz import curried as c
from sklearn import decomposition,datasets
from matplotlib import pyplot as plt


#%%
# k近傍法
X=np.random.rand(10,2)
plt.scatter(*X.T)


#%%
# pointごとの距離を算出
diff=X[np.newaxis,:,:]-X[:,np.newaxis,:]
print(diff.shape)
dis_sq=np.sum(diff**2,axis=-1)
nearest=np.argsort(dis_sq)
nearest

#%%
# 可視化
#K個の最近傍点に興味があるとする
K=2
#K+1個の近傍のindexを取得する(自身が入るのでK+1)
nearest_part=np.argpartition(dis_sq,K+1,axis=1)

nearest

#%%
# 可視化
plt.scatter(*X.T)

for i in range(X.shape[0]):
    for j in nearest_part[i,:K+1]:
        # 二点間の距離をplot
        plt.plot(*zip(X[i],X[j]))


