# %%
from collections import defaultdict
import itertools as it
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#%%
pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
gt = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

def general_confusion_matrix(prd,gt):
    c=np.zeros((2,2))
    for i,j in zip(pred,gt):
        c[i,j]+=1

    return c

general_confusion_matrix(pred,gt)

#%%
# COO形式を利用して、confusion martixを作成
from scipy import sparse
def confusion_matrix(pred,gt):
    cont=sparse.coo_matrix((np.ones(pred.size),(pred,gt)))
    return cont

cont=confusion_matrix(pred,gt)
print(cont)
print('==============')
print(cont.toarray())
