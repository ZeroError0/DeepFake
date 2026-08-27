import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VideoCNN(nn.Module):
    """CNN for deepfake video detection using transfer learning (ResNet18).

    Input: RGB video frame (3, 224, 224)
    Output: binary classification (real vs fake)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # Use pretrained ResNet18 as feature extractor
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Freeze early layers for fine-tuning (train only last ~30%)
        layers_to_freeze = list(self.backbone.children())[:6]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs


class VideoFrameAggregator(nn.Module):
    """Aggregates predictions from multiple frames of a video clip.

    Takes multiple frame predictions and returns a single clip-level prediction.
    """

    def __init__(self, method: str = "mean"):
        super().__init__()
        self.method = method

    def forward(self, frame_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_logits: (batch_frames, num_classes) - logits for each frame
        Returns:
            clip_logits: (1, num_classes) - aggregated clip prediction
        """
        if self.method == "mean":
            aggregated = frame_logits.mean(dim=0, keepdim=True)
        elif self.method == "max":
            aggregated = frame_logits.max(dim=0, keepdim=True).values
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
        return aggregated
