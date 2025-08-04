import torch
from src.Modules import commons
from src import utils
from src.Voice_Synthesizer import SynthesizerTrn
from src.Text.Symbols import symbols
from src.Text import text_to_sequence
from scipy.io.wavfile import write
import logging
import os
import onnxruntime
import numpy as np

# Desactiva los molestos logs de matplotlib si lo usas en otro lado
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class TTS:
    """
    Clase unificada para Texto a Voz (TTS) que soporta tanto modelos
    PyTorch (.pth) como ONNX (.onnx).
    """
    def __init__(self, config_path, model_path, device="cuda"):
        """
        Inicializa el motor TTS. Detecta automáticamente el tipo de modelo.

        Args:
            config_path (str): Ruta al archivo de configuración JSON.
            model_path (str): Ruta al archivo del modelo (.pth o .onnx).
            device (str): Dispositivo a usar ("cuda" o "cpu").
        """
        self.device = device
        self.hps = utils.get_hparams_from_file(config_path)
        self.model_path = model_path
        self.model_type = "onnx" if model_path.endswith(".onnx") else "pytorch"

        self.net_g = None
        self.onnx_session = None

        if self.model_type == "pytorch":
            self._init_pytorch_model()
        else:
            self._init_onnx_model()

        print(f"Motor TTS inicializado en modo: {self.model_type.upper()}")

    def _init_pytorch_model(self):
        """Inicializa el modelo usando PyTorch."""
        if (
            "use_mel_posterior_encoder" in self.hps.model
            and self.hps.model.use_mel_posterior_encoder
        ):
            posterior_channels = 80
        else:
            posterior_channels = self.hps.data.filter_length // 2 + 1

        self.net_g = SynthesizerTrn(
            len(symbols),
            posterior_channels,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        ).to(self.device)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(self.model_path, self.net_g, None)

    def _init_onnx_model(self):
        """Inicializa el motor de inferencia usando ONNX Runtime."""
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        self.onnx_session = onnxruntime.InferenceSession(
            self.model_path, providers=providers
        )

    def _get_text(self, text):
        """Convierte texto plano a una secuencia de IDs de fonemas."""
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return np.array(text_norm, dtype=np.int64)

    def text_to_speech(self, text, output_path="sample.wav", noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0, sid=None):
        """Sintetiza audio a partir de un texto y lo guarda en un archivo WAV."""
        phoneme_ids = self._get_text(text)
        
        if self.model_type == "pytorch":
            # Inferencia con PyTorch
            stn_tst = torch.LongTensor(phoneme_ids).to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(1)]).to(self.device)
            sid_tensor = torch.LongTensor([sid]).to(self.device) if sid is not None else None

            with torch.no_grad():
                audio = self.net_g.infer(
                    stn_tst,
                    x_tst_lengths,
                    sid=sid_tensor,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                )[0][0, 0].data.cpu().float().numpy()

        else: # Inferencia con ONNX
            text_input = np.expand_dims(phoneme_ids, 0)
            text_lengths = np.array([text_input.shape[1]], dtype=np.int64)
            scales = np.array([noise_scale, length_scale, noise_scale_w], dtype=np.float32)
            sid_input = np.array([sid], dtype=np.int64) if sid is not None else None

            audio = self.onnx_session.run(
                None,
                {
                    "input": text_input,
                    "input_lengths": text_lengths,
                    "scales": scales,
                    "sid": sid_input,
                },
            )[0].squeeze((0, 1))

        write(data=audio, rate=self.hps.data.sampling_rate, filename=output_path)
        print(f"Audio guardado exitosamente en: {output_path}")


# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Rutas de configuración
    CONFIG_PATH = "./configs/config.json"
    
    # --- PRUEBA CON MODELO ONNX CUANTIZADO (RECOMENDADO) ---
    ONNX_MODEL_PATH = "./models/LJspeech_quantized.onnx"
    if os.path.exists(ONNX_MODEL_PATH):
        print("\n--- Probando con el modelo ONNX ---")
        tts_onnx_engine = TTS(config_path=CONFIG_PATH, model_path=ONNX_MODEL_PATH)
        tts_onnx_engine.text_to_speech(
            "This is a test using the optimized ONNX model. It should be very fast.",
            "sample_onnx.wav"
        )
    else:
        print(f"No se encontró el modelo ONNX en {ONNX_MODEL_PATH}. Saltando prueba.")

    # --- PRUEBA CON MODELO PYTORCH ORIGINAL ---
    PTH_MODEL_PATH = "./models/LJspeech.pth"
    if os.path.exists(PTH_MODEL_PATH):
        print("\n--- Probando con el modelo PyTorch ---")
        tts_pytorch_engine = TTS(config_path=CONFIG_PATH, model_path=PTH_MODEL_PATH)
        tts_pytorch_engine.text_to_speech(
            "This is a test using the original PyTorch model.",
            "sample_pytorch.wav"
        )
    else:
        print(f"No se encontró el modelo PyTorch en {PTH_MODEL_PATH}. Saltando prueba.")