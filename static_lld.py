# %%
# Static features LLD
import pandas as pd
import numpy as np
import json

df = pd.read_json('lld_feats.json')
df[2] = df[2].apply(lambda x: np.mean(x, axis=0))
df.to_json('static_feats.json', orient="values")

# %%
stat = pd.read_json('static_feats.json')
stat.head()
# %%
foo = stat.iloc[75][2]
print(type(foo))
bar = np.array(foo)
print(type(bar))
print(bar.shape)
# %%
foo = df.iloc[75][2]
print(type(foo))
bar = np.array(foo)
print(type(bar))
print(bar.shape)

# %%
doo = np.apply_along_axis(np.mean(boo, axis=0))

# %%
# df[2].map(lambda x: np.average(x, axis=0))
# print(df[2].head())
# print(df.head())
# %%
foo = df.iloc[0][2]
print(type(foo))
bar = np.array(foo)
print(type(bar))
print(bar.shape)
# %%
new = np.apply_along_axis(lambda x: np.average(x, axis=0), axis=0, arr=boo)

# %%
# %%
# import random
# randomlist = []
# for i in range(0,10):
# n = random.randint(1,10)
# one = randomlist.append(n)
# two = randomlist.append(n)
# # %%
# toy = np.random.randint(0, 10, size=(5, 10, 25))
# toy_df = pd.DataFrame(toy)
# lst1 = [11, 2, 3, 4, 4, 0, 0, ]
# # %%
# # new = np.apply_along_axis(lambda x: np.average(x, axis=0), data)
# print(new.shape)
# # %%# %%
