# %%
# Extract 10 sec segments from long files

from pydub import AudioSegment 
from pydub.utils import make_chunks
import os

def segment_audio(file_name, target_directory):
    myaudio = AudioSegment.from_file(file_name, "wav") 
    chunk_length_ms = 10000 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of 10 sec 
    for i, chunk in enumerate(chunks): 
        # chunk_name = '.../chunked/' + file_name + "chunk{0}.wav".format(i) 
        chunk_name = target_directory + file_name + "chunk{0}.wav".format(i)
        print ("exporting", chunk_name) 
        chunk.export(chunk_name, format="wav") 

# os.makedirs('.../chunked')
def segment_wav_directory(source_directory, target_directory):
    all_file_names = os.listdir(source_directory)
    for file_name in all_file_names:
        if ('.wav' in file_name):
            segment_audio(file_name, target_directory)

source_directory = './audio_test/'
target_directory = './chunked/'
segment_wav_directory(source_directory, target_directory)
# %%
