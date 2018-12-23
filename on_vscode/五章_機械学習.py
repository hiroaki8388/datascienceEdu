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

#%%
# SVM
# merginを最大化するような分類を行う


#%%
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
random_state=0, cluster_std=0.60) 
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');

#%%
from sklearn.svm import SVC
# SVM(線形カーネル)
model=SVC(kernel='linear',C=1E10)
model.fit(X,y)


#%%
# 可視化
def plot_svc_decision_func(model,ax=None,plot_support=True):
    if ax is None:
        ax=plt.gca()
    
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    # 評価モデルのグリッド生成
    x=np.linspace(*xlim,30)
    y=np.linspace(*ylim,30)
    Y,X=np.meshgrid(y,x)

    xy=np.vstack([X.ravel(),Y.ravel()]).T


    P=model.decision_function(xy).reshape(X.shape)

    # 境界とのマージンをplot
    ax.contour(X,Y,P,colors='k',levels=[-1,0,1],alpha=0.5,linstyles=['--','-','--'])

    if plot_support:
        ax.scatter(*model.support_vectors_.T,s=300,lw=1,facecolors='none',edgecolors='black')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
#%%
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
plot_svc_decision_func(model)







#%%
# カーネルSVM
from sklearn.datasets.samples_generator import make_circles
X,y=make_circles(100,factor=.1,noise=.1)
plt.scatter(*X.T,c=y,cmap='autumn')

#%%
# どの基底関数で分離可能かがわかっていれば、基底関数でデータの方を加工すれば良い
from mpl_toolkits import mplot3d

def plot_3D(elev=30,azim=30,X=X,y=y):
    ax=plt.subplot(projection='3d')
    ax.scatter(*X.T,np.exp(-X**2).sum(1),c=y,cmap='autumn')
    ax.view_init(elev=elev,azim=azim)



#%%
# 可視化
# from ipywidgets import interact,fixed
# interact(plot_3D,elev=[30,60],azip=(-180,180),X=fixed(X),y=fixed(y))
plot_3D()

#%%
# RBFカーネルのSVMを適用
clf=SVC(kernel='rbf',C=1E6)
clf.fit(X,y)

#%%
# 結果を可視化
plt.scatter(*X.T,c=y,s=50,cmap='autumn')
plot_svc_decision_func(clf)
# 多項分布NaiveBayes
# ラベルが密以上の場合に使用
from sklearn.datasets import fetch_20newsgroups

categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',
'comp.graphics']
train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)

train['data'][0]


#%%
# tf-idf変換を行い、モデルに入力する
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model=make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(train.data,train.target)

label=model.predict(test.data)

#%%
# from sklearn.metrics import confusion_matrix
# mat=confusion_matrix(test.target,label)
from scipy.sparse import coo_matrix
c_mat=coo_matrix((np.broadcast_to(1,len(label)),(test.target,label)))
c_mat.toarray()

#%%
# 可視化
sns.heatmap(c_mat.toarray().T,square=True,xticklabels=train.target_names,
yticklabels=test.target_names,fmt='d',
annot=True,cbar=False,cmap='RdBu')
#%%
# 基底関数回帰
# 非線形関係にある変数間に線形回帰を適用する

from sklearn.base import BaseEstimator,TransformerMixin

class GaussianFeatures(BaseEstimator,TransformerMixin):

    def __init__(self,N,with_factor=2.0):
        self.N=N
        self.with_factor=with_factor

    @staticmethod
    def _gauss_basis(x,y,width,axis=None):
        args=(x-y)/width 
        return np.exp(-0.5*np.sum(args**2,axis))

    def fit(self,X,y=None):
        self.centers_=np.linspace(X.min(),X.max(),self.N)
        self.with_=self.with_factor*(self.centers_[1]-self.centers_[0])

        return self

    def transform(self,X):
        return self._gauss_basis(X[:,:,np.newaxis],self.centers_,self.with_,axis=1)
#%%
from sklearn.linear_model import LinearRegression
rng = np.random.RandomState(1) 
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

gauss_model=make_pipeline(GaussianFeatures(20),LinearRegression())
gauss_model.fit(x[:,np.newaxis],y)
xfit = np.linspace(0, 10, 1000)
yfit = gauss_model.predict(xfit[:, np.newaxis])
#%%
# 可視化
plt.scatter(x,y)
plt.plot(xfit,yfit)

#%%
