# %%
from collections import defaultdict
import itertools as it
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
style_path = '~/Develop/project/DIUnit/pycoon/dataset/elegant-scipy/style/elegant.mplstyle'
# plt.style.use(style_path)

# %%
# 分位数正規化を行い、任意のデータをある分布に従うように強制する


def quantlie_norm(X):
    """Xの各列をすべて同じ分布に従わせる
    """
    # 基準となる分位数
    quantlie = np.mean(np.sort(X, axis=0), axis=1)

    # 観測データを列ごとにランク付け
    ranks = np.apply_along_axis(
        func1d=stats.rankdata,
        axis=0,
        arr=X
    )

    rank_indices = ranks.astype(int)-1

    # 分位点数に変換
    Xn = quantlie[rank_indices]

    return Xn


def quantlie_norm_log(X):
    """分位数正規化する前にlogに変換する
    """
    logX = np.log10(X+1)
    logXn = quantlie_norm(logX)

    return logXn


# %%
# 遺伝子ごとの、皮膚がん細胞の標本データの読み込み
sample_f_path = '~/Develop/project/DIUnit/pycoon/dataset/elegant-scipy/data/counts.txt'

# 標本である患者のid(column)にどれ位の量の遺伝子(index)が起因しているか
data_tables = pd.read_csv(sample_f_path, index_col=0)
# 標本名(メタデータ)
samples = data_tables.index

counts = data_tables.values


data_tables.head()

# %%
def plot_col_density(data, ax):
    """列ごとにkdeをplotする
    """

    x = np.linspace(np.min(data), np.max(data), 100)
    density_per_col = [
        stats.gaussian_kde(col) for col in data.T
    ]

    for density in density_per_col:
        ax.plot(x=x, y=density(x))
    plt.show()


#%%
# 正規化を行う前の結果
fig, ax = plt.subplots(2,1)
log_counts = np.log(counts+1)
# plot_col_density(log_counts, ax[0])

# 正規化後
log_counts_normalized=quantlie_norm_log(counts)
plot_col_density(log_counts_normalized,ax[1])

#%%
# データのクラスタリングのための値を取得
def most_variable_rows(data,*,n=500):
    """最もばらつきが大きい行を部分集合として取る"""

    rowvar=np.var(data,axis=1)
    sort_indices=np.argsort(rowvar)[-n:]
    variable_data=data[sort_indices,:]

    return variable_data


