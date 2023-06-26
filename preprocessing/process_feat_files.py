
import pandas as pd
import json

# load json feats data
dataset = '.../lld_feats.json' 
df = pd.read_json(dataset)
df.columns = ['AD', 'MMSE', 'LLD']

# create new MMSE only json file for regression
df_MMSE = df.drop(['AD'], axis = 1)
df_MMSE
df_MMSE.to_json('.../MMSE_lld_feats.json', orient="values") 

# create new MMSE only json file for regression
df_AD = df.drop(['MMSE'], axis = 1)
df_AD
df_AD.to_json('.../AD_lld_feats.json', orient="values") 

# create subset AD file for quick debugging, num_rows = 20
df_AD.head(20)
df_AD.head(20).to_json('.../subset20_lld_feats.json', orient="values") 
