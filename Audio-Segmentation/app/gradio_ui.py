import gradio as gr
import os, sys, base64
from pathlib import Path
from core.segmentation import process_audio

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root)

import FVoiceTheme

# ================================= ESTILO =================================
#Tema personalizado de F-Voice
fvoice_theme = FVoiceTheme.FVoiceTheme()

# CSS personalizado
css = """
#logo-header {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px 20px;
}

a {
    text-decoration: none;
}

#logo-header img {
    margin-right: 10px;
}
"""

# Icono de F-VOICE
assets_dir = Path(__file__).resolve().parents[2] / "assets"
def imagen_base64(path):
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/webp;base64,{encoded}"

encoded_img = imagen_base64(assets_dir / "logo.webp")

# ==========================================================================

def process_interface(input_audio, output_folder):
    if not os.path.isdir(output_folder):
        return f"Error: La carpeta '{output_folder}' no existe. Por favor revisa la ruta."


with gr.Blocks(theme=fvoice_theme, css=css) as demo:
    #Logo de F-Voice y redireccionamiento al repositorio
    gr.HTML(f"""
    <div id="logo-header">
        <a href="https://github.com/SIAFI-UNAM/F-VOICE" target="_blank">
            <div style="display: flex; align-items: center; gap: 10px;">
                    <img src="{encoded_img}"  width='100' height='100' />
                    <h1 id='F_VOICE_header' style='margin: 0; font-size:50px'>F-VOICE</h1>
            </div>
        </a>
    </div>
    """)

    # Descripción principal
    gr.Markdown("""
    <div style='font-size:18px; line-height:1.6; color:#FFE3D8; padding: 10px 0;'>
    Bienvenido al segmentador de audio de <strong>F‑VOICE</strong>.  
    Arrastra aquí un archivo en formato <em>.mp3</em> o <em>.wav</em>,  
    o selecciona la carpeta de tu dispositivo que contenga los audios.  
    <br><br>
    Una vez cargados, obtendrás:
    <ul>
        <li>Varios fragmentos de audio separados.</li>
        <li>Un archivo <em>.csv</em> con sus transcripciones.</li>
    </ul>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Área principal de drag and drop
            audio_input = gr.File(
                file_count="multiple",
                file_types=[".wav", ".mp3"],
                label="Arrastra o selecciona tu archivo de audio desde aquí",
                elem_classes="upload-box"
            )
            
            # Sección de configuración
            with gr.Row():
                
                output_dir = gr.Textbox(
                    label="Carpeta de salida",
                    value= Path(__file__).resolve().parents[1]/"app"/"resources"/"Segments",
                    interactive=True,
                    info="Si no eliges otra carpeta, se usará la predeterminada."
                )
            
            process_btn = gr.Button("SEGMENTAR AUDIO", variant="primary")

    # Event handlers
    process_btn.click(
        fn=process_interface,
        inputs=[audio_input, output_dir],
        outputs=[]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        favicon_path=None,
        pwa=True
    )