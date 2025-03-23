import gradio as gr


def update(prompt, model):
    return f"Prompt: {prompt} with {model}!"


class FVoiceTheme(gr.themes.Soft):  # Subclase personalizada del tema base
    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.Color(
                c50="#FFE3D8", c100="#E3B6B1", c200="#845162", c300="#522C5D",
                c400="#29104A", c500="#29104A", c600="#522C5D", c700="#522C5D", c800="#845162", c900="#E3B6B1", c950="#FFE3D8"
            )
        )


fvoice_theme = FVoiceTheme()  # Instanciar el tema personalizado

bg_color = ".gradio-container {background: #150016}"

with gr.Blocks(title="F-VOICE", theme=fvoice_theme, css=bg_color) as demo:
    gr.HTML("<img src='./icon.jpg' width='50' height='50'> F-VOICE", elem_id="title")
    with gr.Row():
        with gr.Column():
            inp = gr.TextArea(placeholder="Escribe tu prompt aquí ...", label="Prompt")
            inp1 = gr.Dropdown(["AMAV1", "AMAV2"], label="Modelo")
            btn = gr.Button("Leer")
            out = gr.Audio()
        with gr.Column():
            gr.Markdown(
            """
            ## F-VOICE

            IA TTS que, a partir de un adio y del texto asociado a este, replicará las características del audio en cada letra.

            Al ingresar un texto nuevo, este será \"leído\" con la voz que se replico, dándole las características que aprendió.
            """, elem_id="description")

    btn.click(fn=update, inputs=[inp, inp1], outputs=out)

demo.launch()
