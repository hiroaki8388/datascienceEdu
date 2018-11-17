#%%
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns


#%% [markdown]
# やること
がんゲノムアトラス(The Cancer Genome Atlas, TCGA)の遺伝子発現データを用 
い、皮膚がん患者の死亡率を予測
#%% [markdown]
#%%
# 遺伝子ごとの、皮膚がん細胞の標本データの読み込み
sample_f_path='~/Develop/project/DIUnit/pycoon/dataset/elegant-scipy/data/counts.txt'

# それぞれの細胞(column)にどれ位の量の遺伝子(index)が起因しているか
data_tables=pd.read_csv(sample_f_path,index_col=0)
# 標本名(メタデータ)
samples=sam_data.index

data_tables.head()
#%%
# 遺伝子ごとのマスターデータ(id,長さ)の読み込み
gene_f_path='~/Develop/project//DIUnit/pycoon/dataset/elegant-scipy/data/genes.csv'
gene_info=pd.read_csv(gene_f_path,index_col=0)
gene_info.head()

#%%
# データの数を揃える
print(ge_data.shape[0])
print(sam_data.shape[0])

