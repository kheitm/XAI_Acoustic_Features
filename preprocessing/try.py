# %%
import pandas as pd
df = pd.read_json ('/Users/kathy-ann/thesis_old/lld_feats.json')
df
# %%
df[0].value_counts()
# %%
