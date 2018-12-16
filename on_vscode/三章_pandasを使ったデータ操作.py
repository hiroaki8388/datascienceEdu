#%%
import numpy as np
import pandas as pd
import scipy as sp
import toolz as tz
from toolz import curried as c
from sklearn import decomposition,datasets
import urllib, gzip,io
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


#%%
# 多重indexをtupleから生成
m_index=pd.MultiIndex.from_tuples(
    [('California', 2000), ('California', 2010), ('New York', 2000), ('New York', 2010),
('Texas', 2000), ('Texas', 2010)]
)
pop=pd.Series(
    data = [33871648, 37253956, 18976457, 19378102, 20851820, 25145561],
    index=m_index
)
# Seriesの中に、二次元的なデータ構造を作れる
pop[:,2010]







#%%
# DataFrameに変換
pop.unstack()
pop[:,2010]

#%%
# MultiIndexの作成
pd.DataFrame(
    np.random.randn(3,3),
    index=[
        list('abc'),
        list('efg')
    ]

)

#%%
# デカルト積からindexを作成
index=pd.MultiIndex.from_product([list('ab'),list('cd')])
pd.DataFrame(
    data=np.random.randn(4,4),
    index=index
)

#%%
# GroupBy
planets=sns.load_dataset('planets')
planets.head()


#%%
planets.groupby('method')['orbital_period'].median()


#%%
# 集約: 適用する関数の返り値はSeries or Scaler
planets.groupby('method')['year'].describe()

#%%
# 集約に複数の関数を適用:返り値はScaler or Series
planets.groupby('method').aggregate(['min',np.median])

#%%
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data1': range(6),
'data2': np.random.randint(0, 10, 6)}, columns = ['key', 'data1', 'data2'])

df.groupby('key').mean()
#%%
# フィルター: 適用する関数の返り値はBooleanのScaler(Bが消えている)
df.groupby('key').filter(lambda x:x['data2'].mean()>4)

#%%
# 変換(適用する関数の返り値は入力と同じ形状)
df.groupby('key').transform(lambda x:x.mean())

#%%
# apply(集約。返り値は任意)
df.groupby('key').apply(lambda x:x.mean())

# apply(変換:返り値はDataFrame)
df.groupby('key').apply(lambda x:x/x['data2'].sum())


#%%
# ピボットテーブル
titanic=sns.load_dataset('titanic')
titanic.pivot_table('survived','sex','class',aggfunc='mean')

#%%
# 多重ピボットテーブル
titanic.pivot_table('survived',['sex',pd.cut(titanic['age'],[0,18,80])],'class')


#%%
titanic.pivot_table('survived',['sex',pd.cut(titanic['age'],[0,18,80])],pd.qcut(titanic['fare'],2))

#%%
# ex 出産データ
births=pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv')
print(births.shape)
births.head()

#%%
# 10年ごと
births['decade']=10*(births['year']//10)
births.head()

#%%
# シグマクリップで外れ値を除外
# sp.stats.sigmaclip(births['births'])
# mean(c) - std(c)*low < c < mean(c) + std(c)*high
quantiles=np.percentile(births['births'],[25,50,75])
mu=quantiles[1]
sig=0.74*(quantiles[2]-quantiles[0])
births=births.query(
    '(births>@mu-5*@sig) & (births<@mu+5*@sig)'
)
births.shape
#%%
# indexを日付に変換
births.index=pd.to_datetime(10000*births['year']+100*births['month']+births['day'],format='%Y%m%d')
births['dayofweek']=births.index.dayofweek
#%%
# 10年ごとの出生率の男女比
births.pivot_table('births','year','gender',aggfunc='sum').plot()

#%%
# 10年ごとの曜日ごとの出産率
births.pivot_table('births','dayofweek','decade',aggfunc='mean').plot()


#%%
# ex シアトル市の自転車数を可視化
data=pd.read_csv('https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD')

data.head()
#%%
# データ処理
data=data.set_index('Date')
data.index=pd.to_datetime(data.index)
data.columns=['West','East']
data['Total']=data.eval('West+East')


#%%
data.dropna().describe()


#%%
# 単純に可視化して見にくい
data.plot()

#%%
# 週ごとに集計
weekly=data.resample('W').sum()
weekly.plot(style=[':','--','-'])


#%%
# 30日ごとの移動平均で集計
daily=data.resample('D').sum()
fig,ax=plt.subplots(2,1)
# 直線的な窓
daily.rolling(30,center=True).sum().plot(ax=ax[0],style=[':','--','-'])

# なめらかにするため、ガウス窓で集計
daily.rolling(30,center=True,win_type='gaussian').sum(std=10).plot(ax=ax[1],style=[':','--','-'])

#%%
# 一日あたりの通行量の可視化
by_time=data.groupby(data.index.time).mean()
hourly_ticks=4*60*60*np.arange(6)
by_time.plot(xticks=hourly_ticks,style=[':','--','-'])