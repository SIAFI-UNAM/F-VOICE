import whisper
import os
from tqdm import tqdm  
import tqdm as tqdm

def voice_to_text(inpath,outpath,model = "base",language = "english"):
    """
    Voice to text function: This function will take audio files and convert it into texts, 
    and save it in a txt file.

    Parameters:
    ----------
    inpath : str
        Path to the folder where the audio files are located.
    outpath : str
        Path to the folder where the txt files will be saved.
    model : str
        The model to be used. It can be either 'base' or 'small'.
    language : str
        The language to be used. It can be either 'english' or 'spanish'.
    """
    print("Starting...")
    # Charge the pre-trained model
    model = whisper.load_model(model)
    model.language = language
    # list for saving results
    results = []
    # Iterate over all the files in the directory
    for file_i in tqdm.tqdm(os.listdir(inpath),desc="Processing voice to text"):
        print(file_i)
        # verify that file is not a directory
        if os.path.isfile(os.path.join(inpath, file_i)):
            # obtain the complete path
            complete_path = os.path.join(inpath, file_i)
            print(complete_path)
            # load and trim audio to just 30 seconds
            audio = whisper.load_audio(complete_path)
            audio = whisper.pad_or_trim(audio)

            # creating mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # Decodify audio
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)

            # append results to the list
            results.append(f"{inpath}/{file_i} | {result.text}")
            # Saving results as a txt file
    with open( outpath+ "/wavs_text.txt", "w") as file:
        for result in results:
            file.write(result + "\n")