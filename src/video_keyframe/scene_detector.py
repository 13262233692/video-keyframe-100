import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Deque
from collections import deque
from .video_decoder import Frame


@dataclass
class SceneChange:
    frame_index: int
    timestamp: float
    score: float
    is_transition: bool = False


@dataclass
class FrameQuality:
    is_blurry: bool
    sharpness_score: float
    brightness_mean: float


class TransitionDetector:
    def __init__(self, window_size: int = 5, transition_threshold: float = 0.15):
        self.window_size = window_size
        self.transition_threshold = transition_threshold
        self.recent_scores: Deque[float] = deque(maxlen=window_size)
        self.in_transition = False
        self.transition_start = -1
        self.transition_frame_count = 0

    def is_in_transition(self, current_score: float, frame_index: int) -> tuple:
        self.recent_scores.append(current_score)
        
        if len(self.recent_scores) < self.window_size:
            return False, 0.0
        
        scores = list(self.recent_scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        is_gradual = (
            mean_score >= self.transition_threshold and
            std_score < 0.15 and
            all(s >= self.transition_threshold * 0.5 for s in scores)
        )
        
        if is_gradual and not self.in_transition:
            self.in_transition = True
            self.transition_start = frame_index
            self.transition_frame_count = 0
        elif is_gradual and self.in_transition:
            self.transition_frame_count += 1
        elif not is_gradual and self.in_transition:
            self.in_transition = False
            transition_duration = frame_index - self.transition_start
            self.transition_frame_count = 0
            return False, transition_duration
        
        return self.in_transition, mean_score


def compute_blur_score(frame: Frame) -> float:
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def compute_frame_quality(frame: Frame, blur_threshold: float = 100.0) -> FrameQuality:
    sharpness = compute_blur_score(frame)
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    return FrameQuality(
        is_blurry=sharpness < blur_threshold,
        sharpness_score=sharpness,
        brightness_mean=brightness
    )


def compute_image_hash(frame: Frame, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.int8)


def compute_hash_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    if hash1 is None or hash2 is None:
        return 0.0
    hamming_distance = np.sum(hash1 != hash2)
    similarity = 1.0 - (hamming_distance / len(hash1))
    return similarity


class HSVSceneDetector:
    def __init__(
        self,
        threshold: float = 0.3,
        hist_bins: int = 50,
        method: str = "correlation"
    ):
        self.threshold = threshold
        self.hist_bins = hist_bins
        self.method = method
        self._last_hist: Optional[np.ndarray] = None

    def _compute_hsv_histogram(self, frame: Frame) -> np.ndarray:
        hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
        
        h_bins = self.hist_bins
        s_bins = self.hist_bins // 2
        v_bins = self.hist_bins // 2
        
        hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
        
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
        
        return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        if self.method == "correlation":
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return 1.0 - max(0, score)
        elif self.method == "chisqr":
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            return min(1.0, score / 10.0)
        elif self.method == "bhattacharyya":
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        elif self.method == "intersection":
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
            return 1.0 - (score / np.sum(hist1))
        else:
            raise ValueError(f"Unknown comparison method: {self.method}")

    def detect_change(self, frame: Frame) -> Optional[SceneChange]:
        current_hist = self._compute_hsv_histogram(frame)
        
        if self._last_hist is None:
            self._last_hist = current_hist
            return None
        
        score = self._compare_histograms(self._last_hist, current_hist)
        self._last_hist = current_hist
        
        if score >= self.threshold:
            return SceneChange(
                frame_index=frame.index,
                timestamp=frame.timestamp,
                score=score
            )
        
        return None

    def reset(self):
        self._last_hist = None


class PixelDiffSceneDetector:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self._last_frame: Optional[np.ndarray] = None

    def detect_change(self, frame: Frame) -> Optional[SceneChange]:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self._last_frame is None:
            self._last_frame = gray
            return None
        
        frame_diff = cv2.absdiff(self._last_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        change_ratio = np.sum(thresh > 0) / (frame.width * frame.height)
        
        self._last_frame = gray
        
        if change_ratio >= self.threshold:
            return SceneChange(
                frame_index=frame.index,
                timestamp=frame.timestamp,
                score=change_ratio
            )
        
        return None

    def reset(self):
        self._last_frame = None


class CombinedSceneDetector:
    def __init__(
        self,
        hsv_threshold: float = 0.3,
        pixel_threshold: float = 0.15,
        use_both: bool = True
    ):
        self.hsv_detector = HSVSceneDetector(threshold=hsv_threshold)
        self.pixel_detector = PixelDiffSceneDetector(threshold=pixel_threshold)
        self.use_both = use_both

    def detect_change(self, frame: Frame) -> Optional[SceneChange]:
        hsv_change = self.hsv_detector.detect_change(frame)
        pixel_change = self.pixel_detector.detect_change(frame)
        
        if self.use_both:
            if hsv_change and pixel_change:
                return SceneChange(
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                    score=(hsv_change.score + pixel_change.score) / 2
                )
        else:
            if hsv_change or pixel_change:
                score = 0.0
                if hsv_change:
                    score = max(score, hsv_change.score)
                if pixel_change:
                    score = max(score, pixel_change.score)
                return SceneChange(
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                    score=score
                )
        
        return None

    def reset(self):
        self.hsv_detector.reset()
        self.pixel_detector.reset()
