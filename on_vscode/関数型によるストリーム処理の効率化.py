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
        c.partition(batch_size),
        c.map(np.array),
        c.map(ipca.partial_fit),
        tz.last
    )

    return ipca

    
#%%
# データを読み取る
file_='../dataset/elegant-scipy/data/iris.csv'

reader=pd.read_csv(file_,chunksize=50)

pca_obj=tz.pipe(
    reader,
    streaming_pca
)

pca_obj


#%%
rand=np.random.randint(1,10,10)

rand


#%%

list(c.partition(4,rand ))


#%%
