# %%
import librosa
import os
import shutil

def measure_duration(chunk_name):
    y, sr = librosa.load(chunk_name)
    dur = librosa.get_duration(y=y, sr=sr)
    return dur

def remove_short_files(source_directory, target_directory):
    all_file_names = os.listdir(source_directory)
    for file_name in all_file_names:
        if ('.wav' in file_name):
            dur = measure_duration(file_name)
            if dur <10.0:
                shutil.move(os.path.join(source_directory, file_name), target_directory)

                
source_directory = '/Users/kathy-ann/thesis/chunked/'
target_directory = '/Users/kathy-ann/thesis/short_files/'
remove_short_files(source_directory, target_directory)

# %%
