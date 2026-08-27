from .models.audio_cnn import AudioCNN
from .models.video_cnn import VideoCNN
from .predict import DeepfakeDetector

__all__ = ["AudioCNN", "VideoCNN", "DeepfakeDetector"]
