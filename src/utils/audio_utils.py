import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
TARGET_LENGTH = 256  # time steps


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Load and resample audio to target sample rate.

    Returns:
        waveform: (1, num_samples)
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    target_length: int = TARGET_LENGTH,
    n_mels: int = N_MELS,
) -> torch.Tensor:
    """Extract mel spectrogram from waveform.

    Args:
        waveform: (1, num_samples)
        target_length: fixed time steps for spectrogram

    Returns:
        mel_spec: (1, n_mels, target_length)
    """
    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
    )
    amp_to_db = AmplitudeToDB(stype="power", top_db=80)

    mel_spec = mel_transform(waveform)
    mel_spec = amp_to_db(mel_spec)

    # Pad or truncate to target length
    if mel_spec.shape[-1] < target_length:
        padding = target_length - mel_spec.shape[-1]
        mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
    else:
        mel_spec = mel_spec[..., :target_length]

    # Normalize to [0, 1]
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

    return mel_spec


def audio_file_to_tensor(path: str) -> torch.Tensor:
    """Full pipeline: load audio file -> mel spectrogram tensor.

    Returns:
        (1, 1, n_mels, target_length) ready for model input
    """
    waveform = load_audio(path)
    mel_spec = extract_mel_spectrogram(waveform)
    return mel_spec.unsqueeze(0)  # add batch dim
