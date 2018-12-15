#%%
import sklearn as sk
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

#%%
# 次元削減
from sklearn.datasets import load_digits
digits=load_digits()

plt.imshow(digits['images'][0],cmap='binary')




#%%
# 多様体学習により、次元を削減する
from sklearn.manifold import Isomap
iso=Isomap(n_components=2)

data_projected=iso.fit_transform(digits.data)

#%%
# 結果を可視化
plt.scatter(*data_projected.T,c=digits.target,cmap=plt.cm.get_cmap('nipy_spectral',10))
plt.colorbar(label='digit label',ticks=range(10))


#%%
# GaussianNaiveBayes
from sklearn.datasets import make_blobs
X,y=make_blobs(100,2,centers=2,cluster_std=1.5,random_state=2)
plt.scatter(*X.T,c=y,cmap='RdBu',s=50)

#%%
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X,y)

X_new=[-6,-14]+[14,18]*np.random.rand(2000,2)
y_new=model.predict(X_new)


#%%
# 可視化
plt.scatter(*X.T,c=y,cmap='RdBu')
lim=plt.axis()
plt.scatter(*X_new.T,c=y_new,cmap='RdBu',alpha=0.1)
plt.axis(lim)