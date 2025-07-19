import numpy as np
import librosa
import soundfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from transformers import pipeline
import csv
import torch

# ================================= PATHS =================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(BASE_DIR, "Resources", "RawAudio", "sample.wav")
output_folder = os.path.join(BASE_DIR, "Resources", "Segments")
# =========================================================================

def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)

class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue

            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        if len(sil_tags) == 0:
            return [[waveform,0,int(total_frames*self.hop_size)]]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append([self._apply_slice(waveform, 0, sil_tags[0][0]),0,int(sil_tags[0][0]*self.hop_size)])
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    [self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]),int(sil_tags[i][1]*self.hop_size),int(sil_tags[i + 1][0]*self.hop_size)]
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    [self._apply_slice(waveform, sil_tags[-1][1], total_frames),int(sil_tags[-1][1]*self.hop_size),int(total_frames*self.hop_size)]
                )
            return chunks
        
def initialize_asr_pipeline(device="cpu"):
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
    )

def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device="cpu")
    
    if isinstance(ref_audio, str):
        audio_array, sr = librosa.load(ref_audio, sr=16000, mono=True)
    else:
        audio_array = ref_audio

    return asr_pipe(
        audio_array,
        chunk_length_s=10,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": "spanish"} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


def generate_metadata(output_folder):
    metadata_path = os.path.join(output_folder, "metadata.csv")
    wav_files = []
    for filename in os.listdir(output_folder):
        if filename.endswith(".wav"):
            parts = filename.rsplit("_", 1)
            number = int(parts[1].split(".")[0])
            wav_files.append((number, filename))
    
    wav_files.sort(key=lambda x: x[0])
    
    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        
        for number, filename in wav_files:
            file_id = os.path.splitext(filename)[0]
            txt_path = os.path.join(output_folder, f"{file_id}.txt")
            
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcription = f.read().strip()
            else:
                transcription = "ERROR_EN_TRANSCRIPCION"
            
            writer.writerow([file_id, transcription, transcription])

    print(f"Metadata generada en: {metadata_path}")

def process_audio(input_file, output_folder, language="spanish"):
    initialize_asr_pipeline(device="cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(input_file, "name"):
        input_path = input_file.name
    else:
        input_path = input_file
    
    audio, sr = librosa.load(input_path, sr=None, mono=False)

    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=200,
        hop_size=10,
        max_sil_kept=500
    )
    
    chunks = slicer.slice(audio)
    print(f"Segmentos creados: {len(chunks)}")
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(input_path).rsplit('.', maxsplit=1)[0]
    segment_paths = []

    for i, (chunk, start, end) in enumerate(chunks):
        if hasattr(chunk, "shape") and len(chunk.shape) > 1:
            chunk = chunk.T
        wav_path = os.path.join(output_folder, f"{base_name}_{i}.wav")
        soundfile.write(wav_path, chunk, sr)
        segment_paths.append(wav_path)

    transcriptions = {}
    for i, wav_path in enumerate(segment_paths):
        try:
            transcription = transcribe(wav_path, language)
            txt_path = wav_path.replace(".wav", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            transcriptions[f"{base_name}_{i}"] = transcription
        except Exception as e:
            print(f"Error al transcribir el segmento {i}: {str(e)}")
    
    generate_metadata(output_folder)
    print(f"Transcripciones completadas")
    return transcriptions

if __name__ == "__main__":
    process_audio(input_path, output_folder, language="spanish")