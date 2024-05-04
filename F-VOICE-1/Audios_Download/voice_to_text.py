import whisper
import os
from tqdm import tqdm

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
    # Charge the pre-trained model
    model = whisper.load_model(model)
    model.language = language
    # list for saving results
    results = []
    # Iterate over all the files in the directory
    for root, dirs, files in os.walk(inpath):
        for file_i in tqdm(files):
            # Verifica que el archivo sea un archivo de audio
            if file_i.endswith(('.wav', '.mp3')):  # Añade aquí otros formatos de audio si es necesario
                complete_path = os.path.join(root, file_i)
                # load and trim audio to just 30 seconds
                audio = whisper.load_audio(complete_path)
                audio = whisper.pad_or_trim(audio)

                # creating mel spectrogram
                mel = whisper.log_mel_spectrogram(audio).to(model.device)

                # Decodify audio
                options = whisper.DecodingOptions()
                result = whisper.decode(model, mel, options)

                # append results to the list
                results.append(f"{file_i} | {result.text}")
                
    # Saving results as a txt file
    with open( outpath+ "/wavs_text.txt", "w") as file:
        for result in results:
            file.write(result + "\n")
