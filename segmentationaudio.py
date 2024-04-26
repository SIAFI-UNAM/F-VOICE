# Import necessary libreries
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

#Function to segment audio
def segment_audii(input_audio_path, output_folder, s=10):
  # Check if s is between 10 and 15
    if s < 10 or s > 15:
        print("Error: La duración de los segmentos debe estar entre 10 y 15 segundos.")
        return False
    #Load the audio file
    audio = AudioSegment.from_file(input_audio_path)
    # Calculate the number of segments
    lon_a = len(audio)
    miliseconds_u = s * 1000
    segments = lon_a / miliseconds_u
    #Create the output folder
    os.makedirs(output_folder, exist_ok=True)
    for i in range(int(segments)):
        audio_i = audio[i*miliseconds_u:miliseconds_u*(i+1)]
        audio_i.export(os.path.join(output_folder, f"segment_{i}.wav"), format="wav")
        #Remove background noise from the segments
    for i in range(int(segments)):
        audio_segment = AudioSegment.from_file(os.path.join(output_folder, f"segment_{i}.wav"))
        audio_segment = audio_segment.low_pass_filter(3000)
        audio_segment.export(os.path.join(output_folder, f"segment_{i}.wav"), format="wav")
        # Print a success message
    print("La operación fue realizada con éxito.")
    return output_folder

#Function to display user options
def user_options():
    print("Por favor, elige una opción:")
    print("1) Segmentar todos los videos de la lista")
    print("2) Elegir mediante número cuál de todos los videos segmentar")
    print("3) Salir del programa")
    option = input("Ingresa el número de tu opción: ")
    return option
# Call the user_options and segment_audii functions according to the user's choice'
while True:
    option = user_options()
    if option == "1":
        video_count = 1
        for filename in os.listdir("Audios Canal"):
            if filename.endswith(".wav"):
                input_audio_path = os.path.join("Audios Canal", filename)
                output_folder = os.path.join("segmented_audios", filename[:-4])  # Create a folder for each video
                while True:
                    s = int(input(f"Para el video {video_count}, ¿cuántos segundos debe durar cada segmento (entre 10 y 15)? "))
                    if s >= 10 and s <= 15:
                        if segment_audii(input_audio_path, output_folder, s):
                            break
                    else:
                        print("Por favor, ingresa un valor entre 10 y 15.")
                video_count += 1
    elif option == "2":
        files = [f for f in os.listdir("Audios Canal") if f.endswith(".wav")]
        for i, filename in enumerate(files):
            print(f"{i+1}) {filename}")
        file_option = int(input("Elige el número del audio que quieres segmentar: "))
        input_audio_path = os.path.join("Audios Canal", files[file_option-1])
        output_folder = os.path.join("segmented_audios", files[file_option-1][:-4])  # Create a folder for the chosen video
        while True:
            s = int(input("¿Cuántos segundos debe durar cada segmento (entre 10 y 15)? "))
            if s >= 10 and s <= 15:
                if segment_audii(input_audio_path, output_folder, s):
                    break
            else:
                print("Por favor, ingresa un valor entre 10 y 15.")
    elif option == "3":
        print("Saliendo del programa.")
        break
    else:
        print("Opción no válida. Por favor, intenta de nuevo.")
