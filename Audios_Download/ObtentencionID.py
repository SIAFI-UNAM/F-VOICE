from pytube import YouTube
import os
import requests
import xml.etree.ElementTree as ET

def IdChannel(id, No_Links): # Recive a string and one number
    id_canal = id  # id "UCFJBsityvF0Z9dfxecDfm7g"
    urlVideo = f'https://www.youtube.com/feeds/videos.xml?channel_id={id_canal}' #Obtenemos el ID del canal --CannelIds":["UCFJBsityvF0Z9dfxecDfm7g"],"p--
    r = requests.get(urlVideo, headers = {"User-Agent": "Chrome/50.0.2661.94"}) # Usamsos el agente o buscador el cual es Chrome para encontar el canal
    data = r.content
    root = ET.fromstring(data.decode()) # Decodificamos el resulatdo de nuestra busqueda
    data.decode()

    lista = root.findall('.//') # Aqui mantenemos las rutas obtenidas, en una lista

    lista

    listaVideos = [] # Creamos una lista la cual va almacenar los links de los videos
    
    cont = 0
    
    for i in lista: # Recorremos nuestra lista
        
        titulo = i.find('{http://search.yahoo.com/mrss/}title') # Para asi obtener los titulos de los videos del canal
        if titulo is None:
            pass # Si el titilo es nulo pasamos y terminamos
        else:
            print('Titulo = ',titulo.text) # En caso contrario imprimmos los titulos y los transormamos a texto

        url = i.get('href') # Para los enlaces hacemos lo mismo pero usamos del xml los href que hace referencia a nuestros links
        if url is None or 'watch' not in url: # Verifica si es nulo el video o si no contiene la palabra watch
            pass
        elif cont <No_Links:
            listaVideos.append(url) #Cuando encuentre los links los añadira a la lista de videos
            cont+=1

    print('Esta es una lista = ',listaVideos)

    for i in listaVideos: # Con el for recorremos nuestra lista de enlaces y los añadimos uno por uno a nuestra funcion 
        yt = YouTube(i) # Obtenemos correctamente el url con la funcion YouTube de pytube
        video = yt.streams.filter(only_audio = True).first() # Aqui indicamos que solo queremos el audio del video
        destino = "Audios Canal"
        out_file = video.download(output_path = destino) # Aqui nosotros inidicamos que el destino sera la carpeteta en la que estamos trabajando
        base, ext = os.path.splitext(out_file) # Aqui generamos el archivo en formato base
        new_file = base + '.wav' # Asignamos el formato a wav
        os.rename(out_file, new_file) # Con esto damos el nuevo formato a nuestro archivo
               
IdChannel("UCFJBsityvF0Z9dfxecDfm7g",6)