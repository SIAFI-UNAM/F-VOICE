from pydub import AudioSegment
from pydub.silence import split_on_silence

def segment_and_save_audio(input_audio_path, output_folder):
    #Cargar el archivo de audio
    audio = AudioSegment.from_file(input_audio_path)

    #Eliminar el ruido de fondo
    audio = audio.low_pass_filter(3000)

    #Segmentar el audio en tramos de 10 a 15 segundos
    segments = split_onsilence(audio, min_silence_len=3000,silence_tresh=-50,keep_silence=100)

    #Crear carpeta de salida
    os.makedirs(output_folder, exist_ok=True)

    #Guardar los segmentos en la carpeta
    for i, segment in enumerate(segments):
        #verificar la duración del segmento
        if len(segment) >= 10000 and len(segment) <= 15000: #Duración de 10 a 15 segundos
            segment.export(os.path.join(output_folder, f"segment_{i}.wav"), format="wav")
        else:
            print(f"egmento {i} es demasiado corto o largo. Duración: {len(segment)} ms")

    return output_folder

