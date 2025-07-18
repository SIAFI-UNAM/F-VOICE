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
    margin-right: 10px;  /* espacio entre imagen y texto */
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

def process_interface(input_audio, output_folder, db_threshold):
    # Lógica de procesamiento (se conectará posteriormente)
    return "Procesamiento completado. Segmentos generados en: " + output_folder

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
                label="Arrastra tus archivos de audio aquí",
                elem_classes="upload-box"
            )
            
            # Sección de configuración
            with gr.Row():
                input_dir = gr.Textbox(
                    label="Carpeta de entrada predeterminada",
                    value=os.path.join("resources", "RawAudio"),
                    interactive=False
                )
                
                output_dir = gr.Textbox(
                    label="Carpeta de salida",
                    value=os.path.join("resources", "Segments"),
                    interactive=True
                )
            
            threshold_slider = gr.Slider(
                minimum=-60, maximum=-20, value=-40,
                label="Umbral de detección de silencios (dB)",
                step=1
            )
            
            process_btn = gr.Button("GENERAR", variant="primary")

        # Sección de resultados
        with gr.Column(scale=2, elem_classes="result-box"):
            gr.Markdown("### Resultados")
            output_result = gr.File(label="Segmentos generados")
            status_output = gr.Textbox(label="Estado del proceso")
            metadata_preview = gr.Dataframe(
                headers=["Archivo", "Duración", "Texto"],
                label="Vista previa de metadatos"
            )

    # Event handlers
    process_btn.click(
        fn=process_interface,
        inputs=[audio_input, output_dir, threshold_slider],
        outputs=[status_output, output_result, metadata_preview]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        favicon_path=None
    )