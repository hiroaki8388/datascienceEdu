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
# 日別の出産数グラフ
births_by_date=births.pivot_table('births',[births.index.month,births.index.day])
births_by_date.head()
dummy_year=2014 # 可視化のため、架空の日付を指定
births_by_date.index=[pd.datetime(2012,m,d) for (m,d) in births_by_date.index]
births_by_date.plot()

#%%
# ex レシピデータ
# データの読み込み
url='./../dataset/recipeitems-latest.json.gz'
# TODO
urls=[url]
gzopen=tz.curry(gzip.open)
read_json=tz.curry(pd.read_json)

recipe=tz.pipe(
urls,
c.map(gzopen(mode='rt')),
tz.concat,
c.do(print),
c.map(lambda line:line.strip()),
c.map(read_json),
tz.last
)


recipe
