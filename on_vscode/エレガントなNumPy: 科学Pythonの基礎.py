# %%
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
style_path='~/Develop/project/DIUnit/pycoon/dataset/elegant-scipy/style/elegant.mplstyle'
# plt.style.use(style_path)

# %% [markdown]
# やること
#がんゲノムアトラス(The Cancer Genome Atlas, TCGA)の遺伝子発現データを用
#い、皮膚がん患者の死亡率を予測
# %% [markdown]
# %%
# 遺伝子ごとの、皮膚がん細胞の標本データの読み込み
sample_f_path = '~/Develop/project/DIUnit/pycoon/dataset/elegant-scipy/data/counts.txt'

# それぞれの細胞(column)にどれ位の量の遺伝子(index)が起因しているか
data_tables = pd.read_csv(sample_f_path, index_col=0)
# 標本名(メタデータ)
samples = sam_data.index

data_tables.head()
# %%
# 遺伝子ごとのマスターデータ(id,長さ)の読み込み
gene_f_path = '~/Develop/project//DIUnit/pycoon/dataset/elegant-scipy/data/genes.csv'
gene_info = pd.read_csv(gene_f_path, index_col=0)
gene_info.head()

# %%
# データの数を揃える
print(gene_info.shape[0])
print(data_tables.shape[0])

# 双方に存在するものだけに絞る
matched_index = pd.Index.intersection(
    self=data_tables.index, other=gene_info.index)
# 必要なデータを再定義
counts=np.asarray(
     data_tables.loc[matched_index],dtype=int)
gene_names=np.array(matched_index)
gene_lengths=gene_info.loc[matched_index]['GeneLength']

print(gene_lengths.shape[0])
print(counts.shape[0])

#%%
# 正規化
## 標本間の正規化するため、標本ごとのばらつきを可視化する
total_counts=np.sum(counts,axis=0)
total_counts.shape

density=sp.stats.gaussian_kde(total_counts)
x=np.arange(min(total_counts),max(total_counts),10000)

fig,ax=plt.subplots()
ax.plot(x=x,y=density(x))

