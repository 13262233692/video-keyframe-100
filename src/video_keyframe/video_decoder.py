import cv2
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray
    width: int
    height: int


class VideoDecoder:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def read_frame(self) -> Optional[Frame]:
        ret, image = self.cap.read()
        if not ret:
            return None
        
        index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        return Frame(
            index=index,
            timestamp=timestamp,
            image=image,
            width=self.width,
            height=self.height
        )

    def iter_frames(self, sample_interval: int = 1) -> Iterator[Frame]:
        frame_idx = 0
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            
            if frame_idx % sample_interval == 0:
                yield frame
            
            frame_idx += 1

    def get_frame_at(self, timestamp: float) -> Optional[Frame]:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        return self.read_frame()

    def get_info(self) -> dict:
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "path": self.video_path
        }
