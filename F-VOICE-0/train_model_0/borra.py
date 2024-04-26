import whisper
import os
from tqdm import tqdm

inpath ="E:\\wavs"
outpath = "E:\\wavs_text"


    # Charge the pre-trained model
model = whisper.load_model("base")
model.language = "english"
    # list for saving results
results = []
    # Iterate over all the files in the directory
for file_i in tqdm(os.listdir(inpath)):
        # verify that file is not a directory
    if os.path.isfile(os.path.join(inpath, file_i)):
            # obtain the complete path
        complete_path = os.path.join(inpath, file_i)
        print(complete_path)
            # load and trim audio to just 30 seconds
        audio = whisper.load_audio(complete_path)
       