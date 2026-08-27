import cv2
import numpy as np
import torch
from torchvision import transforms


TARGET_SIZE = 224
NUM_FRAMES = 10  # frames to sample per video clip


def extract_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES,
) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video file.

    Returns:
        list of RGB frames as numpy arrays (H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not extract any frames from: {video_path}")

    return frames


def preprocess_frames(
    frames: list[np.ndarray],
    target_size: int = TARGET_SIZE,
) -> torch.Tensor:
    """Preprocess frames for model input.

    Args:
        frames: list of RGB numpy arrays
        target_size: resize dimension

    Returns:
        (num_frames, 3, target_size, target_size) tensor
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tensors = []
    for frame in frames:
        tensor = transform(frame)
        tensors.append(tensor)

    return torch.stack(tensors)


def video_file_to_tensor(
    video_path: str,
    num_frames: int = NUM_FRAMES,
) -> torch.Tensor:
    """Full pipeline: load video -> preprocess frames tensor.

    Returns:
        (num_frames, 3, 224, 224) ready for model input
    """
    frames = extract_frames(video_path, num_frames)
    return preprocess_frames(frames)
