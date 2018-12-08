#%%
import numpy as np
import pandas as pd
import scipy as sp
import toolz as tz
from toolz import curried as c
from sklearn import decomposition,datasets
from matplotlib import pyplot as plt
#%%
# PCAのパイプライン実行

def streaming_pca(samples,n_componetnts=2,batch_size=100):
    ipca=decomposition.IncrementalPCA(
    n_components=n_componetnts,
    batch_size=batch_size
    )

    tz.pipe(
        samples,
        # c.partition(batch_size), # batch_size単位のtupleを構築
        c.map(np.array),
        c.map(ipca.partial_fit),
        tz.last # 配列の最後の要素のみを取得
    )

    return ipca

    
#%%
# データを読み取る
file_='../dataset/elegant-scipy/data/iris.csv'
file_t='../dataset/elegant-scipy/data/iris-target.csv'
pd.read_csv(file_,header=None).head()
#%%
#50 行ごとに読み取り
reader=pd.read_csv(file_,chunksize=50,header=None)

# PCAをfit
array=c.curry(np.array)
pca_obj=tz.pipe(
    reader,
    c.map(array(dtype=int)),
    streaming_pca
)

pca_obj




#%%
# PCAをtransform
reshape=c.curry(np.reshape)
reader=pd.read_csv(file_,chunksize=1,header=None)
components=tz.pipe(
    reader,
    c.map(array),
    c.map(pca_obj.transform),
    np.vstack
)


#%%
# 可視化
t=pd.np.loadtxt(file_t)

# 種類別に可視化できている
plt.scatter(*components.T,c=t)







