import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Deque
from collections import deque
from .video_decoder import VideoDecoder, Frame
from .scene_detector import (
    SceneChange, HSVSceneDetector, CombinedSceneDetector,
    TransitionDetector, compute_frame_quality, compute_image_hash, compute_hash_similarity
)
from .ocr_detector import OCRDetector, FrameText
from .chapter_segmenter import ChapterSegmenter, Chapter


@dataclass
class KeyFrame:
    index: int
    timestamp: float
    score: float
    image_path: str = ""
    thumbnail_path: str = ""
    sharpness: float = 0.0
    is_transition: bool = False


@dataclass
class FrameCandidate:
    frame: Frame
    quality: object
    frame_hash: np.ndarray
    scene_score: float


class KeyFrameSelector:
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        detector_type: str = "hsv",
        threshold: float = 0.3,
        min_scene_length: float = 1.0,
        max_keyframes: int = 100,
        sample_interval: int = 2,
        save_thumbnails: bool = True,
        thumbnail_size: tuple = (320, 180),
        detect_transitions: bool = True,
        transition_threshold: float = 0.12,
        blur_threshold: float = 80.0,
        deduplicate: bool = True,
        duplicate_threshold: float = 0.85,
        select_best_in_scene: bool = True,
        enable_chapters: bool = False,
        ocr_languages: List[str] = None,
        ocr_interval: int = 30,
        min_chapter_duration: float = 10.0,
        chapter_similarity_threshold: float = 0.3,
        use_gpu: bool = True
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.max_keyframes = max_keyframes
        self.sample_interval = sample_interval
        self.save_thumbnails = save_thumbnails
        self.thumbnail_size = thumbnail_size
        self.detect_transitions = detect_transitions
        self.blur_threshold = blur_threshold
        self.deduplicate = deduplicate
        self.duplicate_threshold = duplicate_threshold
        self.select_best_in_scene = select_best_in_scene
        self.enable_chapters = enable_chapters
        self.ocr_languages = ocr_languages or ["ch_sim", "en"]
        self.ocr_interval = ocr_interval
        self.min_chapter_duration = min_chapter_duration
        self.chapter_similarity_threshold = chapter_similarity_threshold
        self.use_gpu = use_gpu
        
        if detector_type == "hsv":
            self.detector = HSVSceneDetector(threshold=threshold)
        elif detector_type == "combined":
            self.detector = CombinedSceneDetector(hsv_threshold=threshold)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        self.transition_detector = TransitionDetector(
            window_size=5,
            transition_threshold=transition_threshold
        )
        
        self.ocr_detector: Optional[OCRDetector] = None
        self.chapter_segmenter: Optional[ChapterSegmenter] = None
        if self.enable_chapters:
            self.ocr_detector = OCRDetector(languages=self.ocr_languages, gpu=self.use_gpu)
            if self.ocr_detector.is_available():
                self.chapter_segmenter = ChapterSegmenter(
                    min_chapter_duration=self.min_chapter_duration,
                    similarity_threshold=self.chapter_similarity_threshold
                )
        
        os.makedirs(output_dir, exist_ok=True)
        self.frames_dir = os.path.join(output_dir, "frames")
        self.thumbs_dir = os.path.join(output_dir, "thumbnails")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.thumbs_dir, exist_ok=True)
        
        self.scene_candidates: Deque[FrameCandidate] = deque(maxlen=30)
        self.last_keyframe_hash: Optional[np.ndarray] = None
        self.current_scene_best: Optional[FrameCandidate] = None
        self.frames_since_last_keyframe = 0
        
        self.ocr_results: List[FrameText] = []
        self.scene_change_times: List[float] = []
        self.chapters: List[Chapter] = []

    def _save_frame(self, frame: Frame, prefix: str = "frame") -> tuple:
        filename = f"{prefix}_{frame.index:06d}.jpg"
        frame_path = os.path.join(self.frames_dir, filename)
        cv2.imwrite(frame_path, frame.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        thumb_path = ""
        if self.save_thumbnails:
            thumb = cv2.resize(frame.image, self.thumbnail_size)
            thumb_path = os.path.join(self.thumbs_dir, filename)
            cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        return frame_path, thumb_path

    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _is_duplicate(self, frame_hash: np.ndarray) -> bool:
        if not self.deduplicate or self.last_keyframe_hash is None:
            return False
        similarity = compute_hash_similarity(self.last_keyframe_hash, frame_hash)
        return similarity >= self.duplicate_threshold

    def _select_best_candidate(self) -> Optional[FrameCandidate]:
        if not self.scene_candidates:
            return None
        
        sorted_candidates = sorted(
            self.scene_candidates,
            key=lambda c: (c.quality.sharpness_score, c.scene_score),
            reverse=True
        )
        
        top_count = min(5, len(sorted_candidates))
        top_candidates = sorted_candidates[:top_count]
        
        best = max(top_candidates, key=lambda c: c.frame.index)
        
        return best

    def select_keyframes(self, progress_callback=None) -> List[KeyFrame]:
        keyframes: List[KeyFrame] = []
        skipped_transitions = 0
        skipped_blurry = 0
        skipped_duplicates = 0
        frames_since_ocr = 0
        
        with VideoDecoder(self.video_path) as decoder:
            self.video_info = decoder.get_info()
            last_scene_time = -self.min_scene_length
            first_frame = None
            in_transition = False
            scene_started = False
            
            frames_since_last_keyframe = 0
            recent_scores = deque(maxlen=10)
            scene_change_detected = False
            frames_since_change = 0
            
            for frame in decoder.iter_frames(sample_interval=self.sample_interval):
                if first_frame is None:
                    first_frame = frame
                
                quality = compute_frame_quality(frame, self.blur_threshold)
                frame_hash = compute_image_hash(frame)
                change = self.detector.detect_change(frame)
                
                current_score = change.score if change else 0.0
                recent_scores.append(current_score)
                
                if self.detect_transitions:
                    in_transition, _ = self.transition_detector.is_in_transition(
                        current_score, frame.index
                    )
                else:
                    in_transition = False
                
                candidate = FrameCandidate(
                    frame=frame,
                    quality=quality,
                    frame_hash=frame_hash,
                    scene_score=current_score
                )
                
                if self.enable_chapters and self.ocr_detector and self.ocr_detector.is_available():
                    frames_since_ocr += 1
                    if frames_since_ocr >= self.ocr_interval:
                        ocr_result = self.ocr_detector.detect_text(
                            frame.image, frame.index, frame.timestamp
                        )
                        if ocr_result.combined_text.strip():
                            self.ocr_results.append(ocr_result)
                        frames_since_ocr = 0
                
                if change and change.score > 0.5:
                    self.scene_change_times.append(frame.timestamp)
                
                if self.detect_transitions:
                    self.scene_candidates.append(candidate)
                    
                    if len(self.scene_candidates) > 20:
                        self.scene_candidates.popleft()
                    
                    if current_score > 0.3 and not scene_change_detected:
                        scene_change_detected = True
                        frames_since_change = 0
                    
                    if scene_change_detected:
                        frames_since_change += 1
                    
                    if scene_change_detected and frames_since_change >= 15 and not in_transition:
                        best_candidate = self._select_best_candidate()
                        if best_candidate:
                            time_ok = (best_candidate.frame.timestamp - last_scene_time) >= self.min_scene_length
                            not_duplicate = not self._is_duplicate(best_candidate.frame_hash)
                            not_blurry = best_candidate.quality.sharpness_score >= self.blur_threshold
                            
                            if time_ok and not_blurry and len(keyframes) < self.max_keyframes:
                                if self.deduplicate and not not_duplicate:
                                    skipped_duplicates += 1
                                else:
                                    frame_path, thumb_path = self._save_frame(best_candidate.frame)
                                    keyframes.append(KeyFrame(
                                        index=best_candidate.frame.index,
                                        timestamp=best_candidate.frame.timestamp,
                                        score=best_candidate.scene_score,
                                        image_path=frame_path,
                                        thumbnail_path=thumb_path,
                                        sharpness=best_candidate.quality.sharpness_score,
                                        is_transition=False
                                    ))
                                    self.last_keyframe_hash = best_candidate.frame_hash
                                    last_scene_time = best_candidate.frame.timestamp
                                    frames_since_last_keyframe = 0
                            elif not not_blurry:
                                skipped_blurry += 1
                        
                        scene_change_detected = False
                        frames_since_change = 0
                        self.scene_candidates.clear()
                    
                    if len(keyframes) == 0 and frames_since_last_keyframe >= 20:
                        best_candidate = self._select_best_candidate()
                        if best_candidate:
                            not_duplicate = not self._is_duplicate(best_candidate.frame_hash)
                            not_blurry = best_candidate.quality.sharpness_score >= self.blur_threshold
                            if not_duplicate and not_blurry:
                                frame_path, thumb_path = self._save_frame(best_candidate.frame)
                                keyframes.append(KeyFrame(
                                    index=best_candidate.frame.index,
                                    timestamp=best_candidate.frame.timestamp,
                                    score=best_candidate.scene_score,
                                    image_path=frame_path,
                                    thumbnail_path=thumb_path,
                                    sharpness=best_candidate.quality.sharpness_score,
                                    is_transition=False
                                ))
                                self.last_keyframe_hash = best_candidate.frame_hash
                                last_scene_time = best_candidate.frame.timestamp
                                frames_since_last_keyframe = 0
                                self.scene_candidates.clear()
                            elif not not_blurry:
                                skipped_blurry += 1
                    
                    frames_since_last_keyframe += 1
                else:
                    if change:
                        time_ok = (frame.timestamp - last_scene_time) >= self.min_scene_length
                        not_duplicate = not self._is_duplicate(frame_hash)
                        not_blurry = not quality.is_blurry
                        
                        if time_ok and not_duplicate and not_blurry and len(keyframes) < self.max_keyframes:
                            frame_path, thumb_path = self._save_frame(frame)
                            keyframes.append(KeyFrame(
                                index=frame.index,
                                timestamp=frame.timestamp,
                                score=change.score,
                                image_path=frame_path,
                                thumbnail_path=thumb_path,
                                sharpness=quality.sharpness_score,
                                is_transition=False
                            ))
                            self.last_keyframe_hash = frame_hash
                            last_scene_time = frame.timestamp
                        elif not not_duplicate:
                            skipped_duplicates += 1
                        elif not_blurry:
                            skipped_blurry += 1
                
                if progress_callback:
                    progress = (frame.index / decoder.frame_count) * 100
                    progress_callback(progress, len(keyframes), skipped_transitions)
            
            if self.scene_candidates and len(keyframes) < self.max_keyframes:
                best_candidate = self._select_best_candidate()
                if best_candidate:
                    time_ok = (best_candidate.frame.timestamp - last_scene_time) >= self.min_scene_length
                    not_duplicate = not self._is_duplicate(best_candidate.frame_hash)
                    
                    if time_ok and not_duplicate:
                        frame_path, thumb_path = self._save_frame(best_candidate.frame)
                        keyframes.append(KeyFrame(
                            index=best_candidate.frame.index,
                            timestamp=best_candidate.frame.timestamp,
                            score=best_candidate.scene_score,
                            image_path=frame_path,
                            thumbnail_path=thumb_path,
                            sharpness=best_candidate.quality.sharpness_score,
                            is_transition=False
                        ))
            
            if len(keyframes) == 0 and first_frame is not None:
                frame_path, thumb_path = self._save_frame(first_frame)
                keyframes.append(KeyFrame(
                    index=first_frame.index,
                    timestamp=first_frame.timestamp,
                    score=1.0,
                    image_path=frame_path,
                    thumbnail_path=thumb_path,
                    sharpness=compute_frame_quality(first_frame).sharpness_score
                ))
            
            if self.enable_chapters and self.chapter_segmenter:
                self.chapters = self.chapter_segmenter.segment_chapters(
                    self.ocr_results,
                    self.scene_change_times,
                    self.video_info.get('duration', 0)
                )
                self._assign_keyframes_to_chapters(keyframes)
        
        self.stats = {
            "total_keyframes": len(keyframes),
            "skipped_transitions": skipped_transitions,
            "skipped_blurry": skipped_blurry,
            "skipped_duplicates": skipped_duplicates,
            "chapters_count": len(self.chapters)
        }
        
        return keyframes

    def select_uniform(self, num_frames: int = 10) -> List[KeyFrame]:
        keyframes: List[KeyFrame] = []
        
        with VideoDecoder(self.video_path) as decoder:
            self.video_info = decoder.get_info()
            duration = decoder.duration
            interval = duration / num_frames
            
            for i in range(num_frames):
                timestamp = i * interval
                frame = decoder.get_frame_at(timestamp)
                
                if frame:
                    quality = compute_frame_quality(frame, self.blur_threshold)
                    frame_path, thumb_path = self._save_frame(frame)
                    keyframes.append(KeyFrame(
                        index=frame.index,
                        timestamp=frame.timestamp,
                        score=1.0,
                        image_path=frame_path,
                        thumbnail_path=thumb_path,
                        sharpness=quality.sharpness_score
                    ))
        
        self.stats = {"total_keyframes": len(keyframes)}
        return keyframes

    def _assign_keyframes_to_chapters(self, keyframes: List[KeyFrame]):
        if not self.chapters:
            return
        
        for chapter in self.chapters:
            chapter.keyframes = []
        
        for kf in keyframes:
            chapter = ChapterSegmenter.find_chapter_for_timestamp(self.chapters, kf.timestamp)
            if chapter:
                chapter.keyframes.append(kf)
    
    def get_chapters(self) -> List[Chapter]:
        return self.chapters
    
    def get_video_info(self) -> dict:
        if not hasattr(self, 'video_info'):
            with VideoDecoder(self.video_path) as decoder:
                self.video_info = decoder.get_info()
        return self.video_info

    def get_stats(self) -> dict:
        return getattr(self, 'stats', {})
