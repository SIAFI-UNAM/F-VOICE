
from unidecode import unidecode
import torch
import numpy as np
from utils import load_wav_to_torch
from math import e
from tqdm import tqdm  
#from tqdm import tqdm_notebook as tqdm # Legacy Notebook TQDM
#from tqdm.notebook import tqdm # Modern Notebook TQDM
from shutil import copytree
import matplotlib.pylab as plt
n_gpus=1
rank=0
group_name=None
from scipy.signal import get_window
from librosa.util import pad_center
import torch.nn as nn
from torch.autograd import Variable
import glob
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_decompression

avs = glob.glob('wavs/*.wav')

for i in range(len(avs)):
    audio, sampling_rate = load_wav_to_torch(avs[i])

    ################################
    # Audio Parameters             #
    ################################
    max_wav_value=32768.0
    sampling_rate=22050
    filter_length=1024
    hop_length=256
    win_length=1024
    n_mel_channels=80
    mel_fmin=0.0
    mel_fmax=8000.0

    window='hann'
    forward_transform = None
    scale = filter_length / hop_length

    fourier_basis = np.fft.fft(np.eye(win_length))
    cutoff = int((filter_length / 2 +1))
    fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                    np.imag(fourier_basis[:cutoff, :])])
    forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
    forward_basis
    inverse_basis = torch.FloatTensor(
                np.linalg.pinv(scale * fourier_basis).T[:, None, :])
    if window is not None:
        assert(filter_length >= win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, win_length, fftbins=True)
        fft_window = pad_center(data=fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()
        # window the bases
        forward_basis *= fft_window
        inverse_basis *= fft_window
    T = nn.Module()
    T.register_buffer('forward_basis', forward_basis.float())
    T.register_buffer('inverse_basis', inverse_basis.float())



    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

    audio_norm

    audio_norm2 = audio_norm

    num_batches = audio_norm.size(0)
    num_samples = audio_norm.size(1)
    # similar to librosa, reflect-pad the input
    audio_norm = audio_norm.view(num_batches, 1, num_samples)
    audio_norm = F.pad(
                audio_norm.unsqueeze(1),
                (int(filter_length / 2), int(filter_length / 2), 0, 0),
                mode='reflect')
    audio_norm = audio_norm.squeeze(1)

    forward_transform = F.conv1d(
                audio_norm,
                Variable(forward_basis, requires_grad=False),
                stride=hop_length,
                padding=0)

    cutoff = int((filter_length / 2) + 1)
    real_part = forward_transform[:, :cutoff, :]
    imag_part = forward_transform[:, cutoff:, :]

    magnitude = torch.sqrt(real_part**2 + imag_part**2)
    phase = torch.autograd.Variable(
                torch.atan2(imag_part.data, real_part.data))


    assert(torch.min(audio_norm2.data) >= -1)
    assert(torch.max(audio_norm2.data) <= 1)
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length,n_mels= n_mel_channels,fmin= mel_fmin,fmax=  mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()
    T.register_buffer('mel_basis', mel_basis)
    mel_output = torch.matmul(mel_basis, magnitude)
    mel_output = dynamic_range_decompression(magnitude)
    mel_output = torch.squeeze(mel_output, 0).cpu().numpy()
    print(mel_output)
    ruta_archivo_npy = 'Test/tacotron2-master/tacotron2-master/wavs_mel' + avs[i].replace('.wav', '') + '.npy'
    np.save(avs[i].replace('.wav', ''), mel_output)
    print(i)
