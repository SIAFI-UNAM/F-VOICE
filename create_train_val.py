import os
import random
import argparse

def process_lines(raw_lines, audio_path_prefix):
    """
    Procesa una lista de líneas para darles el formato correcto.
    Formato de salida: ruta/completa/audio.wav|texto_seleccionado
    """
    corrected_lines = []
    print(f"Procesando {len(raw_lines)} líneas...")
    for line in raw_lines:
        # Ignorar líneas en blanco
        if not line.strip():
            continue
        
        parts = line.strip().split('|')
        
        filename = ""
        text = ""
        
        if len(parts) == 2:
            # Caso: nombre.wav|texto
            filename = parts[0]
            text = parts[1]
        elif len(parts) == 3:
            # Caso: nombre.wav|texto_raw|texto_normalizado
            # Se queda con el texto de la derecha (el normalizado)
            filename = parts[0]
            text = parts[2]
        else:
            print(f"⚠️ ADVERTENCIA: Línea ignorada por formato inesperado: {line.strip()}")
            continue
            
        # Construir la ruta completa del audio
        # os.path.join es más robusto para crear rutas de archivo
        full_audio_path = os.path.join(audio_path_prefix, f"{filename}.wav")
        
        # Crear la nueva línea con el formato correcto
        new_line = f"{full_audio_path}|{text}\n"
        corrected_lines.append(new_line)
        
    return corrected_lines

def crear_conjuntos_entrenamiento(metadata_path, train_output, val_output, audio_path, val_split=0.2):
    """
    Lee, procesa, mezcla y divide un archivo de metadatos en conjuntos de entrenamiento y validación.
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Error: El archivo no fue encontrado en la ruta '{metadata_path}'")
        return

    # 1. Procesar las líneas para darles el formato correcto
    processed_lines = process_lines(raw_lines, audio_path)

    # 2. Mezclar aleatoriamente las líneas ya procesadas
    random.shuffle(processed_lines)

    # 3. Dividir en conjuntos de entrenamiento y validación
    num_val = int(len(processed_lines) * val_split)
    val_lines = processed_lines[:num_val]
    train_lines = processed_lines[num_val:]

    # 4. Guardar los archivos
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(val_output), exist_ok=True)

    with open(train_output, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(val_output, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"\n✅ Proceso completado.")
    print(f"   - Muestras de entrenamiento: {len(train_lines)}")
    print(f"   - Muestras de validación:   {len(val_lines)}")
    print(f"   - Archivo de entrenamiento guardado en: {train_output}")
    print(f"   - Archivo de validación guardado en:    {val_output}")

# --- Bloque principal modificado ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesa y divide un archivo de metadatos para entrenamiento de TTS."
    )

    parser.add_argument(
        "metadata_path",
        help="Ruta al archivo de metadatos original (ej. metadata.csv)."
    )
    # --- NUEVO ARGUMENTO OBLIGATORIO ---
    parser.add_argument(
        "--audio_path",
        required=True,
        help="Ruta a la carpeta que contiene los archivos de audio .wav (ej. 'datos/wavs')."
    )
    parser.add_argument(
        "--output_dir",
        default="filelists",
        help="Directorio donde se guardarán los archivos. Por defecto es 'filelists'."
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Proporción para el conjunto de validación. Por defecto es 0.2."
    )

    args = parser.parse_args()

    # --- Construimos las rutas completas usando el directorio de salida ---
    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")

    # Llama a la función principal con el nuevo argumento 'audio_path'
    crear_conjuntos_entrenamiento(
        metadata_path=args.metadata_path,
        train_output=train_path,
        val_output=val_path,
        audio_path=args.audio_path,
        val_split=args.val_split
    )