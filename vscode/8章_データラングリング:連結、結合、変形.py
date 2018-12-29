#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%%
iris = sns.load_dataset('iris')
iris.head()

#%%
iris.set_index(['sepal_length', 'sepal_width'])

#%%
iris.stack()

#%%
iris[(np.abs(iris.iloc[:,:3])>3).any(axis=1)]

#%%
s1=pd.Series([0,1],index=list('ab'))
s2=pd.Series([2,3],index=list('cd'))
d1=pd.DataFrame([[0,1]],index=list('ab'))
d2=pd.DataFrame([[2,3]],index=list('cd'))
pd.concat([s1,s1,s2],keys=['one','two','three'])

#%%
pd.concat([d1,d1,d2],axis=1,keys=['one','two','three'])


#%%
df1 =pd.DataFrame({'a': [1., np.nan, 5., np.nan], 'b': [np.nan, 2., np.nan, 6.],
'c': range(2, 18, 4)})
#%%
df2=pd.DataFrame({'a': [5., 4., np.nan, 3., 7.], 'b': [np.nan, 3., 4., 6., 8.]})



#%%
# 補完
df1.combine_first(df2)

#%%
# pivot
data = pd.DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(['Ohio', 'Colorado'], name='state'),
columns=pd.Index(['one', 'two', 'three'], name='number'))


data

#%%
result = data.stack()
result

#%%
df = pd.DataFrame({'left': result,'right':result+5},columns=pd.Index(['left','right'],name='side'))
df

#%%
df.stack()

#%%
url ='https://raw.githubusercontent.com/wesm/pydata-book/2nd-edition/examples/macrodata.csv'
data=pd.read_csv(url)
data.head()
#%%
periods = pd.PeriodIndex(year = data.year, quarter = data.quarter, name='date')
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D','end')
ldata = data.stack().reset_index().rename(columns = {0:'value'})
ldata.head()

#%%
pivoted=ldata.pivot('date','item','value')
pivoted.head()


#%%
# pivotの逆操作melt
df =  pd.DataFrame({'key': ['foo', 'bar', 'baz'], 'A': [1, 2, 3],'B': [4, 5, 6], 'C': [7, 8, 9]})
df
#%%
pd.melt(df,['key'])

#%%
pd.melt(df,['key']).pivot('key','variable','value')