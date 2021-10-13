
import pandas as pd
import json

# load json feats data
dataset = '/Users/kathy-ann/thesis_old/lld_feats.json' 
df = pd.read_json(dataset)
df.columns = ['AD', 'MMSE', 'LLD']

# create new MMSE only json file for regression
df_MMSE = df.drop(['AD'], axis = 1)
df_MMSE
df_MMSE.to_json('/Users/kathy-ann/thesis_old/MMSE_lld_feats.json', orient="values") 

# create new MMSE only json file for regression
df_AD = df.drop(['MMSE'], axis = 1)
df_AD
df_AD.to_json('/Users/kathy-ann/thesis_old/AD_lld_feats.json', orient="values") 

# create subset AD file for quick debugging, num_rows = 20
df_AD.head(20)
df_AD.head(20).to_json('/Users/kathy-ann/thesis_old/subset20_lld_feats.json', orient="values") 
