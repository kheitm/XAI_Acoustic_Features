# %%
import pandas as pd

#%%
# load funcs and meta data to one df
meta = '.../funcs_meta1.csv'
funcs = '.../final_funcs1.csv'
meta = pd.read_csv(meta, index_col=0)
funcs = pd.read_csv(funcs).drop(['start', 'end'], axis=1)
df = pd.concat([meta, funcs], axis = 1)


