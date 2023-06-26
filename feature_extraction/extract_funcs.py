import pandas as pd
import numpy as np
import librosa
import opensmile

def extract_egemaps(wav_file):
    audio, sr = librosa.load(wav_file) 
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
        num_channels=1, feature_level=opensmile.FeatureLevel.Functionals,)
    row_data = smile.process_signal(audio, sr) 
    return row_data

def egemaps_dataset(fname):
    features = [] 
    df = pd.read_csv(fname, sep='\t')
    for _, row in df.iterrows():
        wav_file, mmse, label = row[0], row[4], row[5]
        try:
            data = extract_egemaps(wav_file)
            funcs_as_list = data.values.tolist()
            features.append([label, mmse, funcs_as_list])
        except (IOError, ValueError):
             print("An I/O error or a ValueError occurred with file {}".format(wav_file))

    dataset = pd.DataFrame(features, columns=['label', 'mmse', 'functionals']) 
    dataset.to_json('.../funcs_feats.json', orient="values")

    return dataset

fname = '.../data.tsv'
egemaps_dataset(fname)
