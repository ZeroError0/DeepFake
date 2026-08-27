from pathlib import Path

import torch
import torch.nn.functional as F

from .models.audio_cnn import AudioCNN
from .models.video_cnn import VideoCNN, VideoFrameAggregator
from .utils.audio_utils import audio_file_to_tensor
from .utils.video_utils import video_file_to_tensor


class DeepfakeDetector:
    """Unified interface for detecting deepfakes in audio and video."""

    def __init__(self, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.audio_model = None
        self.video_model = None

    def load_audio_model(self, checkpoint_path: str):
        """Load trained audio detection model."""
        self.audio_model = AudioCNN(num_classes=2).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio_model.load_state_dict(checkpoint["model_state_dict"])
        self.audio_model.eval()
        print(f"Audio model loaded from {checkpoint_path}")

    def load_video_model(self, checkpoint_path: str):
        """Load trained video detection model."""
        self.video_model = VideoCNN(num_classes=2, pretrained=False).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.video_model.load_state_dict(checkpoint["model_state_dict"])
        self.video_model.eval()
        print(f"Video model loaded from {checkpoint_path}")

    def predict_audio(self, audio_path: str) -> dict:
        """Predict if audio is real or deepfake.

        Returns:
            dict with 'label', 'confidence', and 'probabilities'
        """
        if self.audio_model is None:
            raise RuntimeError("Audio model not loaded. Call load_audio_model() first.")

        tensor = audio_file_to_tensor(audio_path).to(self.device)
        probs = self.audio_model.predict_proba(tensor)

        prob_real = probs[0, 0].item()
        prob_fake = probs[0, 1].item()
        label = "fake" if prob_fake > prob_real else "real"
        confidence = max(prob_real, prob_fake)

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {"real": prob_real, "fake": prob_fake},
        }

    def predict_video(self, video_path: str, num_frames: int = 10) -> dict:
        """Predict if video is real or deepfake.

        Returns:
            dict with 'label', 'confidence', 'probabilities', and 'frame_results'
        """
        if self.video_model is None:
            raise RuntimeError("Video model not loaded. Call load_video_model() first.")

        frames_tensor = video_file_to_tensor(video_path, num_frames).to(self.device)

        # Predict on each frame
        frame_results = []
        all_probs = []

        for i in range(frames_tensor.shape[0]):
            frame = frames_tensor[i].unsqueeze(0)
            probs = self.video_model.predict_proba(frame)
            prob_real = probs[0, 0].item()
            prob_fake = probs[0, 1].item()
            frame_results.append({
                "frame": i,
                "real_prob": prob_real,
                "fake_prob": prob_fake,
                "prediction": "fake" if prob_fake > prob_real else "real",
            })
            all_probs.append(probs)

        # Aggregate across frames
        stacked = torch.cat(all_probs, dim=0)
        avg_probs = stacked.mean(dim=0, keepdim=True)

        prob_real = avg_probs[0, 0].item()
        prob_fake = avg_probs[0, 1].item()
        label = "fake" if prob_fake > prob_real else "real"
        confidence = max(prob_real, prob_fake)

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {"real": prob_real, "fake": prob_fake},
            "frame_results": frame_results,
        }

    def predict(self, file_path: str) -> dict:
        """Auto-detect file type and predict.

        Returns:
            dict with prediction results
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        if ext in audio_exts:
            result = self.predict_audio(file_path)
            result["modality"] = "audio"
        elif ext in video_exts:
            result = self.predict_video(file_path)
            result["modality"] = "video"
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <file_path> [--audio-model MODEL] [--video-model MODEL]")
        sys.exit(1)

    detector = DeepfakeDetector()

    # Parse args
    file_path = sys.argv[1]
    audio_model = None
    video_model = None

    if "--audio-model" in sys.argv:
        audio_model = sys.argv[sys.argv.index("--audio-model") + 1]
    if "--video-model" in sys.argv:
        video_model = sys.argv[sys.argv.index("--video-model") + 1]

    # Auto-load models based on file type
    ext = Path(file_path).suffix.lower()
    if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a"} and audio_model:
        detector.load_audio_model(audio_model)
    elif ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"} and video_model:
        detector.load_video_model(video_model)

    result = detector.predict(file_path)
    print(json.dumps(result, indent=2))
