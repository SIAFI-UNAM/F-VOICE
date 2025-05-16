import gradio as gr
from core.segmentation import process_audio

def gradio_interface(input_audio, db_thresh):
    chunks = process_audio(input_audio, "./output", {"threshold": db_thresh})
    return [chunk["audio_path"] for chunk in chunks]

gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Audio(type="filepath"), gr.Slider(-60, -20)],
    outputs=gr.File(label="Segmentos")
).launch()