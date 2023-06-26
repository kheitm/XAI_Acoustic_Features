import pandas as pd
import os
import glob
import regex as re

def make_metadata_file(cgroup, dgroup):
    meta_data_c  = pd.read_csv(cgroup, sep=';')
    meta_data_c['label'] = int(0)
    meta_data_c['mmse'] = meta_data_c['mmse'].str.lstrip()
    meta_data_c['mmse']= meta_data_c['mmse'].replace('NA', 0)
    meta_data_c['mmse'] = meta_data_c['mmse'].astype(int)
    meta_data_c['mmse'].replace(0, meta_data_c['mmse'].mean().astype(int), inplace=True)
    meta_data_c['label'] = int(0)
    meta_data_d  = pd.read_csv(dgroup, sep=';')
    meta_data_d['label'] = int(1)
    frames = [meta_data_c, meta_data_d]
    metadata = pd.concat(frames)
    metadata.columns = ['id', 'age', 'gender', 'mmse', 'label']
    metadata['id'] = metadata['id'].str.rstrip()
    return metadata


def make_data_file(cgroup, dgroup, dir, output):
    metadata = make_metadata_file(cgroup, dgroup)
    os.chdir(dir)
    wav_lst = [file for file in glob.glob('*.wav')]
    row_data =[]
    for w in wav_lst:
        pattern = re.findall(r'^[S][0-9]{3}', w)[0]
        info = metadata.loc[metadata['id']== pattern].values.flatten().tolist()
        info.insert(0, w)
        row_data.append(info)
    data = pd.DataFrame.from_records(row_data)
    data.columns = ['segment', 'id', 'age', 'gender', 'mmse', 'label'] 
    data.to_csv(output, sep = '\t', index=False) 
    return


cgroup = '.../train/cc_meta_data.txt'
dgroup = '.../train/cd_meta_data.txt'
dir = '.../chunked/'
output = '.../data2.tsv'
make_data_file(cgroup, dgroup, dir, output)
