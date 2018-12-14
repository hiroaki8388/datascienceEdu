# https://devlights.hatenablog.com/entry/2018/03/29/123802

#%%
import pandas as pd
import itertools as it
import operator as op

#%%
listA=list('abc')
listB=list(range(1,10))
listC=list(range(100,1000,100))

#%%
list(it.chain(listA,listB))

#%%
for i ,j in it.zip_longest(listA,listB):
    print(f"{i},{j}")
#%%
cycle_iter=it.cycle(listA)

for i,j in zip(listB,cycle_iter):
    print(f"{i},{j}")
#%%
print(list(it.accumulate(listB,op.add)))
print(list(it.accumulate(listB,op.mul)))
print(list(it.accumulate([0,1,2,1,3,-1,10,12,14],max)))
#%%
print(listB)
print(list(it.islice(listB,2)))
print(list(it.islice(listB,2,4)))
print(list(it.islice(listB,2,8,2)))

#%%
list(it.starmap(op.mul,list(zip(listB,listC))))
#%%
print(list(it.product(list('ab'),list('cd'),repeat=2)))
print(list(it.product(pd.date_range(pd.datetime.today(),periods=3,freq='D'),[ 1,2,3])))
#%%
list(it.permutations('ABCD'))
#%%
print(list(it.combinations('ABC',r=2)))
print(list(it.combinations_with_replacement('ABC',r=2)))
#%%
robots = [{
    'name': 'blaster',
    'faction': 'autobot'
}, {
    'name': 'galvatron',
    'faction': 'decepticon'
}, {
    'name': 'jazz',
    'faction': 'autobot'
}, {
    'name': 'metroplex',
    'faction': 'autobot'
}, {
    'name': 'megatron',
    'faction': 'decepticon'
}, {
    'name': 'starcream',
    'faction': 'decepticon'
}]

for k,g in it.groupby(robots,lambda x:x['faction']):
    print(
        f'k:{k},g:{list(g)}\n'
    )









































