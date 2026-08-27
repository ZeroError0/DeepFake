# Deepfake Audio/Video Detector

Binary classification system for detecting AI-generated (deepfake) audio and video content.

## Architecture

```
Audio Path:  WAV/MP3 → Mel Spectrogram (128×256) → 4-layer CNN → Real/Fake
Video Path:  MP4/AVI → Frame Extraction → ResNet18 (transfer learning) → Real/Fake
```

## Project Structure

```
deepfake-detector/
├── src/
│   ├── models/
│   │   ├── audio_cnn.py      # Custom CNN for mel spectrograms
│   │   └── video_cnn.py      # ResNet18-based with transfer learning
│   ├── utils/
│   │   ├── audio_utils.py    # Mel spectrogram extraction
│   │   └── video_utils.py    # Frame extraction & preprocessing
│   ├── dataset.py            # PyTorch Dataset classes
│   └── predict.py            # Unified prediction interface
├── train.py                  # Training script
├── app.py                    # Streamlit demo UI
├── checkpoints/              # Saved model weights
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your data in this structure:

```
data/
├── train/
│   ├── real/
│   │   ├── *.wav (or *.mp4)
│   │   └── ...
│   └── fake/
│       ├── *.wav (or *.mp4)
│       └── ...
└── val/
    ├── real/
    └── fake/
```

**Free Datasets:**
- **Audio**: [ASVspoof](https://www.asvspoof.org/) — direct download, no approval
- **Video**: [DFDC Sample on Kaggle](https://www.kaggle.com/datasets) — ~1-2GB, no approval needed
- **Video**: [FaceForensics++](https://github.com/ondyari/FaceForensics) — fill request form (academic use)

### 3. Train Models

**Audio detector:**
```bash
python train.py --mode audio --train-dir data/train --val-dir data/val --epochs 25
```

**Video detector:**
```bash
python train.py --mode video --train-dir data/train --val-dir data/val --epochs 15 --batch-size 8
```

### 4. Run Inference

```bash
python -m src.predict path/to/file.wav --audio-model checkpoints/best_audio_model.pth
python -m src.predict path/to/file.mp4 --video-model checkpoints/best_video_model.pth
```

### 5. Launch Demo UI

```bash
streamlit run app.py
```

## Model Details

### Audio CNN
- Input: Mel spectrogram (1, 128, 256)
- 4 convolutional blocks with batch norm + dropout
- Adaptive average pooling → FC classifier
- Trained on ASVspoof spectrograms

### Video CNN
- Input: RGB frames (3, 224, 224)
- ResNet18 backbone with pretrained ImageNet weights
- Early layers frozen, fine-tune last ~30%
- Frame-level predictions aggregated via mean pooling

## Requirements

- Python 3.9+
- GPU recommended (Colab free tier works)
- ~2-4GB disk for small datasets

## License

MIT
