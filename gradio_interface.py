import FVoiceTheme

import gradio as gr

from pathlib import Path
import torch

import commons
import utils
import logging
logging.getLogger().setLevel(logging.INFO)  # Eliminacion de logs no deseados

from data_utils import (
    TextAudioLoader,
    TextAudioCollate,
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)

from models import SynthesizerTrn
from text import text_to_sequence


#def inference(device, model, prompt):
def inference(model, prompt):
    # if device == "cuda":
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hps = utils.get_hparams_from_file("./configs/vits2_ama.json")

    net_g = SynthesizerTrn(
        n_vocab=178,
        spec_channels=80, # <--- vits2 parameter (changed from 513 to 80)
        segment_size=8192,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
        n_speakers=0,
        gin_channels=0,
        use_sdp=False, 
        use_transformer_flows=True, # <--- vits2 parameter
        # (choose from "pre_conv", "fft", "mono_layer_inter_residual", "mono_layer_post_residual")
        transformer_flow_type="pre_conv", # <--- vits2 parameter 
        use_spk_conditioned_encoder=True, # <--- vits2 parameter
        use_noise_scaled_mas=True, # <--- vits2 parameter
        use_duration_discriminator=True, # <--- vits2 parameter
    )

    x = torch.LongTensor([[1, 2, 3],[4, 5, 6]]) # token ids
    x_lengths = torch.LongTensor([3, 2]) # token lengths
    y = torch.randn(2, 80, 100) # mel spectrograms
    y_lengths = torch.Tensor([100, 80]) # mel spectrogram lengths

    net_g(
        x=x,
        x_lengths=x_lengths,
        y=y,
        y_lengths=y_lengths,
    )
    _ = net_g.eval()

    _ = utils.load_checkpoint(f"./models/{model}", net_g, None)

    # Mover el modelo a cpu o gpu
    net_g = net_g.to(device)

    # Obtener el texto y moverlo a GPU
    stn_tst = get_text(prompt, hps)
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

    # Inferencia
    with torch.no_grad():
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.75, noise_scale_w=0.8, length_scale=1)[0][0, 0].data.cpu().float().numpy()

    # tupla para el componente de audio de gradio (sample rate, audio data en np.array)
    audio = (hps.data.sampling_rate, audio)
    # Se actualiza el audio y el encabezado de la sección
    return audio, "### Audio Generado"


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# USER INTERFACE

# Instanciacion del tema personalizado
fvoice_theme = FVoiceTheme.FVoiceTheme()

# CSS personalizado
css = """
#logo-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
}

a {
    text-decoration: none;
}
"""

# Ruta de assets (para icono de F-VOICE)
gr.set_static_paths(paths=[Path.cwd().absolute()/"assets"])

# Intefaz con gradio blocks
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

    # Descripción principal
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
            model = gr.Dropdown(["AMA_V3.pth"], label="Modelo")
            # device = gr.Dropdown(["cuda", "cpu"], label="Procesamiento")
            btn = gr.Button("Generar")

    # Encabezado de la seccion del audio
    markdown_output = gr.Markdown("### Ejemplo de voz")

    # Audio (preview y generado)
    audio = gr.Audio(value="assets/preview.wav", autoplay=False, label="Voz reproducida", interactive=False)

    # btn.click(fn=inference, inputs=[device, model, prompt], outputs=[audio, markdown_output])
    btn.click(fn=inference, inputs=[model, prompt], outputs=[audio, markdown_output])

demo.launch()
