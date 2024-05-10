from pytube import YouTube
import os

def download_videos_from_file(file_path, output_folder):
    # Lee los enlaces del archivo de texto
    with open(file_path, 'r') as file:
        video_links = file.readlines()
    for link in video_links:
        try:
            yt = YouTube(link)
            video = yt.streams.filter(only_audio = True).first()
            print(f"Downloading video: {yt.title}...")
            destino = "Audios Canal"
            out_file = video.download(output_path = destino) # Aqui nosotros inidicamos que el destino sera la carpeteta en la que estamos trabajando
            base, ext = os.path.splitext(out_file) # Aqui generamos el archivo en formato base
            new_file = base + '.wav' # Asignamos el formato a wav
            os.rename(out_file, new_file) # Con esto damos el nuevo formato a nuestro archivo
            print("¡Download succes!")
        except Exception as e:
            print(f"Error with {link}: {str(e)}")

    