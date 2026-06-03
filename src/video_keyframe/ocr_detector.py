import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class OCRResult:
    timestamp: float
    frame_index: int
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class FrameText:
    frame_index: int
    timestamp: float
    texts: List[OCRResult] = field(default_factory=list)
    combined_text: str = ""

    def __post_init__(self):
        if self.texts and not self.combined_text:
            self.combined_text = " ".join([t.text for t in self.texts])


class OCRDetector:
    def __init__(self, languages: List[str] = None, gpu: bool = True):
        self.languages = languages or ["ch_sim", "en"]
        self.gpu = gpu
        self._reader = None
        self._init_reader()

    def _init_reader(self):
        try:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        except ImportError:
            print("[WARNING] EasyOCR not installed. OCR functionality will be disabled.")
            self._reader = None
        except Exception as e:
            print(f"[WARNING] Failed to initialize EasyOCR: {e}")
            self._reader = None

    def is_available(self) -> bool:
        return self._reader is not None

    def detect_text(self, image: np.ndarray, frame_index: int, timestamp: float,
                    min_confidence: float = 0.5) -> FrameText:
        if not self._reader:
            return FrameText(frame_index=frame_index, timestamp=timestamp)

        try:
            results = self._reader.readtext(image)
        except Exception:
            return FrameText(frame_index=frame_index, timestamp=timestamp)

        texts = []
        for bbox, text, confidence in results:
            if confidence >= min_confidence and text.strip():
                try:
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x_min, y_min = int(min(x_coords)), int(min(y_coords))
                    x_max, y_max = int(max(x_coords)), int(max(y_coords))
                    
                    texts.append(OCRResult(
                        timestamp=timestamp,
                        frame_index=frame_index,
                        text=text.strip(),
                        confidence=float(confidence),
                        bbox=(x_min, y_min, x_max, y_max)
                    ))
                except (ValueError, TypeError):
                    continue

        return FrameText(
            frame_index=frame_index,
            timestamp=timestamp,
            texts=texts
        )

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary
