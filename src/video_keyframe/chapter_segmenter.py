import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .ocr_detector import FrameText, OCRResult


@dataclass
class Chapter:
    index: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    title: str = ""
    keyframes: List = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    ocr_texts: List[str] = field(default_factory=list)


class ChapterSegmenter:
    def __init__(self, 
                 min_chapter_duration: float = 10.0,
                 similarity_threshold: float = 0.3,
                 max_chapters: int = 20,
                 use_ocr_only: bool = False):
        self.min_chapter_duration = min_chapter_duration
        self.similarity_threshold = similarity_threshold
        self.max_chapters = max_chapters
        self.use_ocr_only = use_ocr_only
        self._vectorizer = None

    @staticmethod
    def preprocess_text(text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                        'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 
                        'were', 'be', 'been', 'being', 'have', 'has', 'had',
                        'do', 'does', 'did', 'will', 'would', 'could', 'should',
                        'may', 'might', 'must', 'can', 'shall', 'will',
                        '的', '了', '和', '与', '及', '或', '在', '是', '有',
                        '为', '这', '那', '个', '就', '都', '而', '及', '与',
                        '我', '你', '他', '她', '它', '我们', '你们', '他们'])
        
        words = text.lower().split()
        words = [w for w in words if w not in stopwords and len(w) > 1]
        
        return ' '.join(words)

    @staticmethod
    def compute_text_similarity(text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        
        t1 = set(text1.split())
        t2 = set(text2.split())
        
        if not t1 or not t2:
            return 0.0
        
        intersection = len(t1 & t2)
        union = len(t1 | t2)
        
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_tfidf_similarity(texts: List[str]) -> np.ndarray:
        if not SKLEARN_AVAILABLE or len(texts) < 2:
            n = len(texts)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    sim_matrix[i, j] = ChapterSegmenter.compute_text_similarity(texts[i], texts[j])
            return sim_matrix
        
        processed_texts = [ChapterSegmenter.preprocess_text(t) for t in texts]
        
        vectorizer = TfidfVectorizer(max_features=500)
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception:
            n = len(texts)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    sim_matrix[i, j] = ChapterSegmenter.compute_text_similarity(texts[i], texts[j])
            return sim_matrix

    @staticmethod
    def detect_text_change_points(ocr_results: List[FrameText], 
                                   window_size: int = 5,
                                   threshold: float = 0.5) -> List[int]:
        if len(ocr_results) < 3:
            return []
        
        texts = [r.combined_text for r in ocr_results]
        similarity_matrix = ChapterSegmenter.compute_tfidf_similarity(texts)
        
        change_points = []
        
        for i in range(1, len(ocr_results) - 1):
            start_prev = max(0, i - window_size)
            end_prev = i
            start_next = i
            end_next = min(len(ocr_results), i + window_size)
            
            if end_prev - start_prev < 2 or end_next - start_next < 2:
                continue
            
            prev_sims = similarity_matrix[start_prev:end_prev, start_prev:end_prev]
            next_sims = similarity_matrix[start_next:end_next, start_next:end_next]
            cross_sims = similarity_matrix[start_prev:end_prev, start_next:end_next]
            
            avg_within = (np.mean(prev_sims) + np.mean(next_sims)) / 2
            avg_cross = np.mean(cross_sims)
            
            change_score = avg_within - avg_cross
            
            if change_score > threshold:
                change_points.append(i)
        
        return change_points

    @staticmethod
    def extract_keywords(texts: List[str], top_k: int = 5) -> List[str]:
        all_words = []
        for text in texts:
            processed = ChapterSegmenter.preprocess_text(text)
            all_words.extend(processed.split())
        
        counter = Counter(all_words)
        keywords = [word for word, count in counter.most_common(top_k) if len(word) > 1]
        return keywords

    @staticmethod
    def generate_title(texts: List[str], duration: float, index: int) -> str:
        keywords = ChapterSegmenter.extract_keywords(texts, top_k=3)
        
        if len(texts) > 0 and texts[0].strip():
            first_text = texts[0].strip()
            if len(first_text) > 50:
                first_text = first_text[:50] + "..."
            if keywords:
                return f"章节 {index + 1}: {first_text[:20]}"
            return f"章节 {index + 1}: {first_text[:30]}"
        
        if keywords:
            return f"章节 {index + 1}: {' '.join(keywords[:2])}"
        
        mins = int(duration // 60)
        secs = int(duration % 60)
        return f"章节 {index + 1} ({mins:02d}:{secs:02d})"

    def segment_chapters(self, 
                          ocr_results: List[FrameText],
                          scene_change_times: Optional[List[float]] = None,
                          video_duration: float = 0) -> List[Chapter]:
        if not ocr_results:
            return [Chapter(
                index=0,
                start_time=0,
                end_time=video_duration,
                start_frame=0,
                end_frame=0,
                title=f"章节 1",
                keywords=[],
                ocr_texts=[]
            )]

        chapters = []
        
        change_points = self.detect_text_change_points(
            ocr_results, 
            window_size=5,
            threshold=self.similarity_threshold
        )
        
        if scene_change_times and not self.use_ocr_only:
            frame_times = [r.timestamp for r in ocr_results]
            for scene_time in scene_change_times:
                closest_idx = min(range(len(frame_times)), 
                                 key=lambda i: abs(frame_times[i] - scene_time))
                if closest_idx not in change_points:
                    change_points.append(closest_idx)
        
        change_points = sorted(set(change_points))
        
        start_idx = 0
        chapter_index = 0
        
        for cp_idx in change_points:
            if cp_idx <= start_idx:
                continue
                
            start_time = ocr_results[start_idx].timestamp
            end_time = ocr_results[cp_idx].timestamp
            duration = end_time - start_time
            
            if duration >= self.min_chapter_duration or chapter_index == 0:
                chapter_texts = [r.combined_text for r in ocr_results[start_idx:cp_idx] 
                               if r.combined_text.strip()]
                
                chapter = Chapter(
                    index=chapter_index,
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=ocr_results[start_idx].frame_index,
                    end_frame=ocr_results[cp_idx].frame_index,
                    keywords=self.extract_keywords(chapter_texts),
                    ocr_texts=chapter_texts
                )
                chapter.title = self.generate_title(chapter_texts, duration, chapter_index)
                chapters.append(chapter)
                chapter_index += 1
                start_idx = cp_idx
        
        if start_idx < len(ocr_results):
            start_time = ocr_results[start_idx].timestamp
            end_time = ocr_results[-1].timestamp if ocr_results else video_duration
            duration = end_time - start_time
            
            chapter_texts = [r.combined_text for r in ocr_results[start_idx:] 
                           if r.combined_text.strip()]
            
            chapter = Chapter(
                index=chapter_index,
                start_time=start_time,
                end_time=end_time,
                start_frame=ocr_results[start_idx].frame_index,
                end_frame=ocr_results[-1].frame_index if ocr_results else 0,
                keywords=self.extract_keywords(chapter_texts),
                ocr_texts=chapter_texts
            )
            chapter.title = self.generate_title(chapter_texts, duration, chapter_index)
            chapters.append(chapter)
        
        if len(chapters) > self.max_chapters:
            chapters = chapters[:self.max_chapters]
        
        if chapters:
            chapters[-1].end_time = video_duration
        
        return chapters

    @staticmethod
    def find_chapter_for_timestamp(chapters: List[Chapter], timestamp: float) -> Optional[Chapter]:
        for chapter in chapters:
            if chapter.start_time <= timestamp <= chapter.end_time:
                return chapter
            if chapter.end_time > timestamp:
                return chapter
        return chapters[-1] if chapters else None
