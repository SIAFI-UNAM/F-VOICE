import gradio as gr
import os
from core.segmentation import process_audio

# ================================= ESTILO =================================
custom_theme = gr.themes.Default(
    primary_hue="emerald",
    secondary_hue="gray",
).set(
    body_background_fill='*neutral_950',
    button_primary_background_fill='*primary_500',
    button_primary_text_color='white',
    border_color_primary='*primary_300',
    slider_color='*primary_400'
)

css = """
footer {visibility: hidden !important}
.upload-box {border: 2px dashed #10b981 !important; padding: 2rem}
.result-box {border-radius: 8px !important; padding: 1.5rem}
"""
# ==========================================================================

def process_interface(input_audio, output_folder, db_threshold):
    # Lógica de procesamiento (se conectará posteriormente)
    return "Procesamiento completado. Segmentos generados en: " + output_folder

with gr.Blocks(theme=custom_theme, css=css) as demo:
    gr.Markdown("# 🎙️ F-VOICE - Segmentador de Audio")
    gr.Markdown("### Carga tus audios y genera segmentos con transcripción")
    
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