#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%%
url = 'https://raw.githubusercontent.com/wesm/pydata-book/2nd-edition/examples/macrodata.csv'
macro = pd.read_csv(url)
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
trans_data.head()

#%%
trans_data.plot()

#%%
sns.regplot('m1', 'unemp', data= trans_data)
#%%
sns.pairplot(trans_data, )

sns.pairplot(trans_data, diag_kind='kde', plot_kws={'alpha':0.2})

#%%
tips = sns.load_dataset('tips')
tips.head()

#%%
# 複数のカテゴリがある場合
sns.factorplot(x='day', y='tip', hue='time', col='smoker',kind='bar', data=tips)

#%%
sns.factorplot(x='day', y='tip', row='time', col='smoker',kind='bar', data=tips)


#%%
sns.factorplot(x='day', y='tip', row='time', col='smoker',kind='box', data=tips)
