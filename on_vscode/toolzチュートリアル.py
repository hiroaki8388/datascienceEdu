#%%
import toolz as tz
from toolz import curried as c
import numpy as np
import pandas as pd

%load_ext memory_profiler
#%%
# データセット
url='http://www.gutenberg.org/files/98/98-0.txt'
book=pd.read_table(url,header=None,chunksize=1)
# mapで遅延評価
str_book=c.map(str,book) # この時点では評価されていない
loud_book=c.map(str.upper,str_book) # この時点では評価されていない

#%%
next(loud_book)

#%%
# pipeを使って構築
book=c.curry( pd.read_table(url,header=None,chunksize=1)
pipe=tz.pipe(book,
            c.map(str),
            c.map(str.upper),
            next
            )
pipe
#%%
# 縮約系の処理
# 単語のカウント
tz.frequencies(tz.concat(loud_book))
#%%
%memit tz.frequencies(tz.concat(list(loud_book)))
#%%
%memit tz.frequencies(tz.concat(loud_book))
#%%
# ストリーム処理
accounts = [(1, 'Alice', 100, 'F'),  # id, name, balance, gender
(2, 'Bob', 200, 'M'),
(3, 'Charlie', 150, 'M'),
(4, 'Dennis', 50, 'M'),
(5, 'Edith', 300, 'F')]

c.pipe(
    accounts,
    c.filter(lambda x:x[2]>150),
    c.map(c.get([1,2])),
    list
)

#%%

# keyでgroupyby
tz.groupby(c.get(3),accounts)

#%%
# groupbyした集合ごとに演算する
iseven=lambda n:n%2==0
add=lambda x,y:x+y
tz.reduceby(iseven,add,np.arange(1000))


#%%
tz.reduceby(c.get(3),lambda total,x:total+x[2],accounts,0)
#%%
# 結合
addresses = [(1, '123 Main Street'),  # id, address
(2, '5 Adams Way'),
(5, '34 Rue St Michel')]
# toolzのjoinは右側のjoinはstreamで行われるので、巨大なデータセットは右側に持っていく
result=tz.join(tz.first,accounts,tz.first,addresses)
list(result)


#%%
# keyを関数で指定
list(tz.join(iseven,[1,2,3,4],iseven,[7,8,9]))
