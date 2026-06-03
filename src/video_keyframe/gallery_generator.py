import os
import json
from pathlib import Path
from jinja2 import Template
from typing import List, Dict, Optional
from .keyframe_selector import KeyFrame
from .chapter_segmenter import Chapter


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Keyframe Gallery - {{ video_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }

        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 2rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: #888;
            font-size: 0.9rem;
        }

        .video-info {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .info-item {
            text-align: center;
        }

        .info-item .label {
            font-size: 0.8rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .info-item .value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #00d4ff;
            margin-top: 0.3rem;
        }

        .chapters-nav {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }

        .chapters-nav h2 {
            font-size: 1.3rem;
            color: #fff;
            margin-bottom: 1rem;
        }

        .chapter-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .chapter-tag {
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }

        .chapter-tag:hover {
            background: rgba(0, 212, 255, 0.1);
            border-color: rgba(0, 212, 255, 0.3);
        }

        .chapter-tag.active {
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            border-color: transparent;
        }

        .gallery-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .chapter-section {
            margin-bottom: 3rem;
        }

        .chapter-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .chapter-title {
            font-size: 1.3rem;
            color: #fff;
        }

        .chapter-meta {
            font-size: 0.85rem;
            color: #888;
        }

        .chapter-keywords {
            margin-top: 0.5rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .keyword-tag {
            padding: 0.2rem 0.6rem;
            background: rgba(124, 58, 237, 0.2);
            border-radius: 12px;
            font-size: 0.75rem;
            color: #a78bfa;
        }

        .gallery-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .gallery-header h2 {
            font-size: 1.3rem;
            color: #fff;
        }

        .keyframe-count {
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            padding: 0.3rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
        }

        .keyframe-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .keyframe-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(0, 212, 255, 0.3);
        }

        .frame-image {
            width: 100%;
            aspect-ratio: 16/9;
            object-fit: cover;
            background: #000;
        }

        .frame-info {
            padding: 1rem;
        }

        .frame-time {
            font-size: 1.1rem;
            font-weight: 600;
            color: #00d4ff;
            margin-bottom: 0.5rem;
        }

        .frame-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: #888;
        }

        .score-bar {
            margin-top: 0.5rem;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
        }

        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            border-radius: 2px;
            transition: width 0.3s ease;
        }

        .lightbox {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .lightbox.active {
            display: flex;
        }

        .lightbox-content {
            max-width: 90%;
            max-height: 90%;
        }

        .lightbox-content img {
            max-width: 100%;
            max-height: 80vh;
            border-radius: 8px;
        }

        .lightbox-info {
            text-align: center;
            margin-top: 1rem;
            color: #fff;
        }

        .lightbox-close {
            position: absolute;
            top: 2rem;
            right: 2rem;
            font-size: 2rem;
            color: #fff;
            cursor: pointer;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }

        .lightbox-close:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .lightbox-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            font-size: 3rem;
            color: #fff;
            cursor: pointer;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            transition: all 0.3s ease;
            user-select: none;
        }

        .lightbox-nav:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .lightbox-prev {
            left: 2rem;
        }

        .lightbox-next {
            right: 2rem;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.85rem;
        }

        .footer a {
            color: #00d4ff;
            text-decoration: none;
        }

        @media (max-width: 768px) {
            .gallery-container {
                padding: 1rem;
            }
            
            .gallery {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 1rem;
            }
            
            .lightbox-nav {
                font-size: 2rem;
                padding: 0.5rem;
            }
            
            .lightbox-prev {
                left: 0.5rem;
            }
            
            .lightbox-next {
                right: 0.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Video Keyframe Gallery</h1>
        <p>{{ video_name }}</p>
    </div>

    <div class="video-info">
        <div class="info-item">
            <div class="label">分辨率</div>
            <div class="value">{{ video_info.width }} × {{ video_info.height }}</div>
        </div>
        <div class="info-item">
            <div class="label">帧率</div>
            <div class="value">{{ "%.1f" | format(video_info.fps) }} FPS</div>
        </div>
        <div class="info-item">
            <div class="label">总帧数</div>
            <div class="value">{{ video_info.frame_count }}</div>
        </div>
        <div class="info-item">
            <div class="label">时长</div>
            <div class="value">{{ duration_formatted }}</div>
        </div>
    </div>

    {% if chapters %}
    <div class="chapters-nav">
        <h2>章节导航</h2>
        <div class="chapter-list">
            {% for chap in chapters %}
            <div class="chapter-tag" onclick="scrollToChapter('chapter-{{ chap.index }}')">
                {{ chap.title }}
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    <div class="gallery-container">
        {% if chapters %}
            {% for chap in chapters %}
            <div class="chapter-section" id="chapter-{{ chap.index }}">
                <div class="chapter-header">
                    <div>
                        <h2 class="chapter-title">{{ chap.title }}</h2>
                        <div class="chapter-meta">
                            {{ chap.start_formatted }} - {{ chap.end_formatted }} | 时长: {{ chap.duration_formatted }}
                        </div>
                        {% if chap.keywords %}
                        <div class="chapter-keywords">
                            {% for kw in chap.keywords[:5] %}
                            <span class="keyword-tag">{{ kw }}</span>
                            {% endfor %}
                        </div>
                        {% endif %}
                    </div>
                    <span class="keyframe-count">{{ chap.keyframes|length }} 帧</span>
                </div>

                <div class="gallery">
                    {% for kf in chap.keyframes %}
                    <div class="keyframe-card" onclick="openLightbox({{ kf.global_index }})">
                        <img class="frame-image" src="{{ kf.thumb_rel }}" alt="Frame {{ kf.index }}">
                        <div class="frame-info">
                            <div class="frame-time">{{ kf.time_formatted }}</div>
                            <div class="frame-meta">
                                <span>帧 #{{ kf.index }}</span>
                                <span>场景变化: {{ "%.1f%%" | format(kf.score * 100) }}</span>
                            </div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: {{ kf.score * 100 }}%"></div>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        {% else %}
        <div class="gallery-header">
            <h2>关键帧列表</h2>
            <span class="keyframe-count">{{ keyframes|length }} 帧</span>
        </div>

        <div class="gallery">
            {% for kf in keyframes %}
            <div class="keyframe-card" onclick="openLightbox({{ loop.index0 }})">
                <img class="frame-image" src="{{ kf.thumb_rel }}" alt="Frame {{ kf.index }}">
                <div class="frame-info">
                    <div class="frame-time">{{ kf.time_formatted }}</div>
                    <div class="frame-meta">
                        <span>帧 #{{ kf.index }}</span>
                        <span>场景变化: {{ "%.1f%%" | format(kf.score * 100) }}</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {{ kf.score * 100 }}%"></div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>

    <div class="lightbox" id="lightbox">
        <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
        <span class="lightbox-nav lightbox-prev" onclick="prevFrame()">&#10094;</span>
        <span class="lightbox-nav lightbox-next" onclick="nextFrame()">&#10095;</span>
        <div class="lightbox-content">
            <img id="lightbox-img" src="" alt="Full size frame">
            <div class="lightbox-info">
                <span id="lightbox-time"></span> | 
                <span id="lightbox-frame"></span> | 
                <span id="lightbox-score"></span>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Generated by <a href="https://github.com/video-keyframe/video-keyframe">Video Keyframe Tool</a></p>
    </div>

    <script>
        const keyframes = {{ keyframes_json | safe }};
        let currentIndex = 0;

        function scrollToChapter(chapterId) {
            const element = document.getElementById(chapterId);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        function openLightbox(index) {
            currentIndex = index;
            updateLightbox();
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeLightbox() {
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }

        function updateLightbox() {
            const kf = keyframes[currentIndex];
            document.getElementById('lightbox-img').src = kf.image_rel;
            document.getElementById('lightbox-time').textContent = kf.time_formatted;
            document.getElementById('lightbox-frame').textContent = 'Frame #' + kf.index;
            document.getElementById('lightbox-score').textContent = 'Score: ' + (kf.score * 100).toFixed(1) + '%';
        }

        function nextFrame() {
            currentIndex = (currentIndex + 1) % keyframes.length;
            updateLightbox();
        }

        function prevFrame() {
            currentIndex = (currentIndex - 1 + keyframes.length) % keyframes.length;
            updateLightbox();
        }

        document.addEventListener('keydown', function(e) {
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowRight') nextFrame();
            if (e.key === 'ArrowLeft') prevFrame();
        });

        document.getElementById('lightbox').addEventListener('click', function(e) {
            if (e.target === this) closeLightbox();
        });
    </script>
</body>
</html>
"""


class GalleryGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _process_keyframe(self, kf: KeyFrame, global_index: int) -> Dict:
        image_rel = os.path.relpath(kf.image_path, self.output_dir).replace('\\', '/')
        thumb_rel = os.path.relpath(kf.thumbnail_path, self.output_dir).replace('\\', '/') if kf.thumbnail_path else image_rel
        
        return {
            "index": kf.index,
            "timestamp": kf.timestamp,
            "time_formatted": self._format_time(kf.timestamp),
            "score": kf.score,
            "image_path": kf.image_path,
            "thumbnail_path": kf.thumbnail_path,
            "image_rel": image_rel,
            "thumb_rel": thumb_rel,
            "global_index": global_index
        }

    def generate(
        self,
        keyframes: List[KeyFrame],
        video_info: Dict,
        video_name: str,
        chapters: Optional[List[Chapter]] = None
    ) -> str:
        output_path = os.path.join(self.output_dir, "index.html")
        
        keyframes_data = []
        global_index_map = {}
        
        for idx, kf in enumerate(keyframes):
            kf_data = self._process_keyframe(kf, idx)
            keyframes_data.append(kf_data)
            global_index_map[kf.index] = idx
        
        chapters_data = None
        if chapters and len(chapters) > 0:
            chapters_data = []
            for chap in chapters:
                chap_keyframes = []
                for kf in chap.keyframes:
                    if kf.index in global_index_map:
                        kf_data = self._process_keyframe(kf, global_index_map[kf.index])
                        chap_keyframes.append(kf_data)
                
                if chap_keyframes:
                    chapters_data.append({
                        "index": chap.index,
                        "title": chap.title,
                        "start_time": chap.start_time,
                        "end_time": chap.end_time,
                        "start_formatted": self._format_time(chap.start_time),
                        "end_formatted": self._format_time(chap.end_time),
                        "duration_formatted": self._format_time(chap.end_time - chap.start_time),
                        "keywords": chap.keywords,
                        "keyframes": chap_keyframes
                    })
        
        template = Template(HTML_TEMPLATE)
        html_content = template.render(
            video_name=video_name,
            video_info=video_info,
            duration_formatted=self._format_time(video_info.get("duration", 0)),
            keyframes=keyframes_data,
            keyframes_json=json.dumps(keyframes_data, ensure_ascii=False),
            chapters=chapters_data
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_path
