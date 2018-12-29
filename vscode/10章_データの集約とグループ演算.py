#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%%
tips = sns.load_dataset('tips')
tips['tip_pct'] = tips['tip'] /tips['total_bill']
tips.head()

#%%
# 要約統計(条件を絞る)
group = tips.groupby(['day', 'smoker'])

group[['tip','total_bill']].agg(['mean', 'sum', 'std',('peak', lambda x:np.max(x)-np.min(x))])

#%%
#apply
tips.groupby(['smoker'])['tip','total_bill'].apply(lambda x:x.head())

#%%
tips.groupby(['smoker'])['tip','total_bill'].describe()


#%%
frame = pd.DataFrame({'data1': np.random.randn(1000),'data2': np.random.randn(1000)})
frame.head()

#%%
quant = pd.cut(frame.data1, 4)
quant.head()

#%%
frame.groupby(quant).max()

#%%
url = 'https://raw.githubusercontent.com/wesm/pydata-book/2nd-edition/examples/stock_px_2.csv'
close_px = pd.read_csv(url, index_col=0, parse_dates=True)
close_px.head()

#%%
# 日時の履歴とSPXとの年次の相関を調べる
rets = close_px.pct_change().dropna()
rets.head()

#%%
spx_corr = lambda x: x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x:x.year)
by_year.agg(spx_corr).plot()

#%%
# 年ごとに線形回帰を行う
import statsmodels.api as sm

def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

by_year.apply(regress, 'AAPL', ['SPX'])

#%%
# pivot_table
tips.head()

#%%
tips.pivot_table(['tip', 'size'], index=['time', 'day'], columns=['smoker'], margins=True)

#%%
# 人数
tips.pivot_table('tip', index=('time', 'day'), columns='smoker', aggfunc=len, fill_value=0, margins=True)

#%%
# レコード数ならcross_tabのほうが良い
pd.crosstab(index = [tips.time, tips.day], columns=tips.smoker, margins=True)
