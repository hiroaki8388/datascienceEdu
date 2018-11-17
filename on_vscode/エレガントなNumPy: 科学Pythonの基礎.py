# %%
from collections import defaultdict
import itertools as it
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
style_path = '~/Develop/project/DIUnit/pycoon/dataset/elegant-scipy/style/elegant.mplstyle'
# plt.style.use(style_path)

# %% [markdown]
# やること
# がんゲノムアトラス(The Cancer Genome Atlas, TCGA)の遺伝子発現データを用
# い、皮膚がん患者の死亡率を予測
# %% [markdown]
# %%
# 遺伝子ごとの、皮膚がん細胞の標本データの読み込み
sample_f_path = '~/Develop/project/DIUnit/pycoon/dataset/elegant-scipy/data/counts.txt'

# 標本である患者のid(column)にどれ位の量の遺伝子(index)が起因しているか
data_tables = pd.read_csv(sample_f_path, index_col=0)
# 標本名(メタデータ)
samples = data_tables.index

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
counts = np.asarray(
    data_tables.loc[matched_index], dtype=int)
gene_names = np.array(matched_index)
gene_lengths = gene_info.loc[matched_index]['GeneLength']

print(gene_lengths.shape[0])
print(counts.shape[0])

# %%
# 正規化
# 標本間の正規化するため、標本ごとのばらつきを可視化する
total_counts = np.sum(counts, axis=0)
total_counts.shape

density = sp.stats.kde.gaussian_kde(total_counts)
x = np.arange(min(total_counts), max(total_counts), 1000)
y = density(x)

# 人によって測定したリード数の総量は全く異なるため、正規化が必要


# %%
# ランダムにサンプリングしてきて、正規化の様子をみる

np.random.seed(7)
sample_index = np.random.choice(counts.shape[1], size=70, replace=False)

counts_subset = counts[:, sample_index]

# ticksを間引くメソッド


def reduce_xaxis_label(ax, factor):
    # 一旦全部消す
    plt.setp(ax.xaxis.get_ticklabels(), visible=False)
    # 一部を可視化する
    for label in ax.xaxis.get_ticklabels()[factor-1::factor]:
        label.set_visible(True)


fig, ax = plt.subplots(3, 1)
ax[0].boxplot(counts_subset)
reduce_xaxis_label(ax[0], 5)

# 対数スケール
ax[1].boxplot(np.log10(counts_subset,))
reduce_xaxis_label(ax[1], 5)
# 正規化+対数スケール
ax[2].boxplot(np.log10(counts_subset/total_counts[sample_index]))
reduce_xaxis_label(ax[2], 5)
plt.show()


# %%
# 正規化前と後を比較

def class_boxplot(data, classes, color=None, **kargs):
    all_classes = sorted(set(classes))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    class2color = dict(zip(all_classes, it.cycle(colors)))

    class2data = defaultdict(list)

    for distrib, cls in zip(data, classes):
        # いったんすべてを初期化
        for c in all_classes:
            class2data[c].append([])
        # 末尾に追加していく
        class2data[cls][-1] = distrib

    fig, ax = plt.subplots()
    lines = []
    for cls in all_classes:
        for key in ['boxprops', 'whiskerprops', 'flierprops']:
            kargs.setdefault(key, {}).update(color=class2color[cls])

        box = ax.boxplot(class2data[cls], **kargs)
        lines.append(box['whiskers'][0])
    ax.legend(lines, all_classes)
    return ax


log_count_3 = list(np.log1p(counts.T[:3]))
counts_lib_norm = counts / total_counts * 1000000
log_ncount_3 = list(np.log1p((counts_lib_norm[:3].T)))
ax = class_boxplot(log_count_3+log_ncount_3,
                    classes=['raw_counts']*3+['norm_counts']*3, labels=[1, 2, 3, 1, 2, 3])
                    
# 明らかにサンプルごとの分散が小さくなる

# %%cyc=it.cycle(list('abc'))
list(zip([12, 3, 1, 3], cyc))
colors = plt.rcParams['axes.prop_cycle']
colors


# %%
