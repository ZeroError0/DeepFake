import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils.audio_utils import load_audio, extract_mel_spectrogram
from .utils.video_utils import extract_frames, preprocess_frames


class AudioDataset(Dataset):
    """Dataset for deepfake audio detection.

    Expected directory structure:
        data_dir/
            real/
                *.wav
            fake/
                *.wav
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        for label, category in enumerate(["real", "fake"]):
            category_dir = self.data_dir / category
            if not category_dir.exists():
                continue
            for ext in ["*.wav", "*.mp3", "*.flac"]:
                for path in category_dir.glob(ext):
                    self.samples.append((str(path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = load_audio(path)
        mel_spec = extract_mel_spectrogram(waveform)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, label


class VideoDataset(Dataset):
    """Dataset for deepfake video detection.

    Expected directory structure:
        data_dir/
            real/
                *.mp4
            fake/
                *.mp4
    """

    def __init__(self, data_dir: str, num_frames: int = 10, transform=None):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []

        for label, category in enumerate(["real", "fake"]):
            category_dir = self.data_dir / category
            if not category_dir.exists():
                continue
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                for path in category_dir.glob(ext):
                    self.samples.append((str(path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = extract_frames(path, self.num_frames)
        frames_tensor = preprocess_frames(frames)

        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, label


def get_audio_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Create train and validation data loaders for audio."""
    train_dataset = AudioDataset(train_dir)
    val_dataset = AudioDataset(val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def get_video_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
):
    """Create train and validation data loaders for video."""
    train_dataset = VideoDataset(train_dir)
    val_dataset = VideoDataset(val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
