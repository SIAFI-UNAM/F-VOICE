import src.FVoiceTheme as FVoiceTheme
import gradio as gr
from pathlib import Path
import os
import time
from src import utils
import logging
import warnings

# --- CONFIGURACIÓN PARA LIMPIAR LA CONSOLA ---

# 1. Suprimir la advertencia de 'weight_norm' de PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Establecer un nivel de registro más alto para las bibliotecas ruidosas
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# 1. Importa tu clase TTS desde el archivo de inferencia
from inference import TTS 

# --- CONFIGURACIÓN Y GESTIÓN DINÁMICA DE MODELOS ---

# Rutas a los directorios de modelos y configuraciones
MODEL_DIR = "./models/"
CONFIG_DIR = "./configs/"

# Crear un directorio temporal para los audios generados
os.makedirs("temp_audio", exist_ok=True)

# Caché para almacenar los motores TTS cargados y evitar recargarlos
tts_engines_cache = {}

def get_available_models():
    """Escanea el directorio de modelos y devuelve una lista de archivos .pth y .onnx."""
    if not os.path.exists(MODEL_DIR):
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth") or f.endswith(".onnx")]

def load_engine(model_name):
    """
    Carga un motor TTS si no está en la caché.
    Busca dinámicamente el archivo de configuración correspondiente.
    """
    if model_name not in tts_engines_cache:
        print(f"Cargando modelo: {model_name}...")
        
        # Construir rutas dinámicamente
        model_path = os.path.join(MODEL_DIR, model_name)
        base_name = os.path.splitext(model_name)[0]
        config_path = os.path.join(CONFIG_DIR, f"{base_name}.json")

        # Verificar que ambos archivos existan
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"No se encontró el archivo de configuración correspondiente: {config_path}")

        # Crear y cachear la nueva instancia del motor con su config específica
        print(f"Usando configuración: {config_path}")
        tts_engines_cache[model_name] = TTS(
            config_path=config_path, 
            model_path=model_path
        )
        print(f"Modelo {model_name} cargado y cacheado.")
    
    return tts_engines_cache[model_name]

# 2. FUNCIÓN DE INFERENCIA DINÁMICA (sin cambios)
def inference(model_name, prompt):
    """
    Carga el modelo seleccionado (si es necesario) y genera el audio.
    """
    if not model_name:
        return None, "Error: Por favor, selecciona un modelo."
    
    try:
        tts_engine = load_engine(model_name)
        output_path = os.path.join("temp_audio", f"audio_{int(time.time())}.wav")
        tts_engine.text_to_speech(prompt, output_path, noise_scale=0.75, noise_scale_w=0.8, length_scale=1)
        return output_path, f"### Audio Generado con {model_name}"
    except Exception as e:
        print(f"Ocurrió un error durante la inferencia: {e}")
        return None, f"Error: {e}"


# --- INTERFAZ DE USUARIO CON GRADIO ---

fvoice_theme = FVoiceTheme.FVoiceTheme()
css = """
#logo-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 20px; }
a { text-decoration: none; }
"""
gr.set_static_paths(paths=[Path.cwd().absolute()/"src/assets"])

available_models = get_available_models()

with gr.Blocks(title="F-VOICE", theme=fvoice_theme, css=css) as demo:
    gr.HTML("""
    <div id="logo-header">
        <a href="https://github.com/SIAFI-UNAM/F-VOICE" target="_blank">
            <div style="display: flex; align-items: center; gap: 10px;">
                <img src='/gradio_api/file=assets/logo.webp' width='100' height='100' />
                <h1 id='F_VOICE_header' style='margin: 0; font-size:50px'>F-VOICE</h1>
            </div>
        </a>
    </div>
    """)
    gr.Markdown("""
    <div style='font-size:18px; line-height:1.6; color:#FFE3D8; padding: 10px 0;'>
    <strong>F-VOICE</strong> es un sistema TTS (Text-to-Speech) que utiliza modelos neuronales avanzados
    para sintetizar audio a partir de texto, replicando características vocales aprendidas.<br><br>
    Al ingresar un texto nuevo, este será <em>"leído"</em> con la voz que se replicó, dándole las características que aprendió.
    </div>
    """)

    with gr.Row():
        with gr.Column():
            prompt = gr.TextArea(placeholder="Escribe tu prompt aquí ...", label="Prompt")
        with gr.Column():
            model = gr.Dropdown(
                available_models, 
                label="Modelo", 
                value=available_models[0] if available_models else None
            )
            btn = gr.Button("Generar")

    markdown_output = gr.Markdown("### Ejemplo de voz")
    audio = gr.Audio(value="assets/preview.wav", autoplay=False, label="Voz reproducida", interactive=False)

    btn.click(fn=inference, inputs=[model, prompt], outputs=[audio, markdown_output])

if __name__ == "__main__":
    print("Recordatorio: Asegúrate de que cada modelo en la carpeta './models' tenga un archivo de configuración .json con el mismo nombre en la carpeta './configs'.")
    if not available_models:
        print("ADVERTENCIA: No se encontraron modelos en la carpeta './models/'. La aplicación se ejecutará pero no podrá generar audio.")
    demo.launch()
