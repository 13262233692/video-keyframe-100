# Video Keyframe - 视频关键帧提取工具

一个基于Python的CLI工具，通过场景检测算法从视频中提取关键帧，并生成精美的静态HTML画廊。

## 功能特点

- 🎬 **视频解码**: 基于OpenCV的高效视频解码
- 🔍 **场景检测**: 支持HSV直方图和像素差异两种检测算法
- 🎯 **关键帧选择**: 智能筛选代表性关键帧
- 🎨 **画廊生成**: 生成响应式HTML画廊，支持灯箱查看
- ⚡ **高性能**: 支持帧采样，处理速度快

## 模块架构

```
video_keyframe/
├── video_decoder.py      # 视频解码模块
├── scene_detector.py     # 场景检测模块
├── keyframe_selector.py  # 关键帧选择模块
├── gallery_generator.py  # 画廊生成模块
└── cli.py               # CLI命令行入口
```

## 安装

```bash
# 克隆或下载项目后，在项目目录执行
pip install -e .
```

## 使用方法

### 1. 查看视频信息

```bash
video-keyframe info video.mp4
```

输出示例：
```
📹 视频信息:
   文件: video.mp4
   分辨率: 1920 x 1080
   帧率: 30.00 FPS
   总帧数: 1800
   时长: 60.00 秒
   格式化时长: 00:01:00
```

### 2. 提取关键帧

```bash
# 基础用法
video-keyframe extract video.mp4

# 指定输出目录
video-keyframe extract video.mp4 -o output_dir

# 调整检测阈值 (0-1，值越小越敏感)
video-keyframe extract video.mp4 -t 0.25

# 设置最小场景时长（避免提取过多相似帧）
video-keyframe extract video.mp4 -m 2.0

# 限制最大关键帧数量
video-keyframe extract video.mp4 --max-keyframes 50

# 均匀采样（不使用场景检测）
video-keyframe extract video.mp4 --uniform 20

# 不生成HTML画廊
video-keyframe extract video.mp4 --no-gallery
```

### 3. 单独生成画廊

如果已经有关键帧图片，可以单独生成HTML画廊：

```bash
video-keyframe gallery keyframes_dir video.mp4
```

## 技术说明

### 场景检测算法

**HSV直方图检测 (`--detector hsv`)**
- 将帧转换为HSV颜色空间
- 计算H/S/V三个通道的直方图
- 使用相关性/巴氏距离等方法比较直方图差异
- 优点：对光照变化鲁棒，适合颜色变化明显的场景

**像素差异检测**
- 灰度化+高斯模糊处理
- 计算帧间像素差异
- 统计变化像素比例
- 优点：计算速度快，适合运动场景

**组合检测 (`--detector combined`)**
- 同时使用两种算法
- 需要两种算法都判定为场景变化时才触发
- 提高检测准确率，减少误报

### 关键帧选择策略

1. 按指定间隔采样帧（默认每2帧采样1次）
2. 检测场景变化点
3. 过滤过短的场景（由`--min-scene-length`控制）
4. 限制最大关键帧数量
5. 自动保存原图和缩略图

### HTML画廊特性

- 响应式网格布局
- 卡片悬停动画效果
- 点击图片进入灯箱模式
- 键盘导航（左右箭头、ESC退出）
- 显示时间戳、帧号、场景变化得分
- 深色主题设计

## 输出目录结构

```
output_dir/
├── frames/           # 原始尺寸关键帧
│   ├── frame_000001.jpg
│   ├── frame_000234.jpg
│   └── ...
├── thumbnails/       # 缩略图 (320x180)
│   ├── frame_000001.jpg
│   ├── frame_000234.jpg
│   └── ...
└── index.html       # HTML画廊页面
```

## API使用示例

除了CLI命令，也可以作为Python库使用：

```python
from video_keyframe.video_decoder import VideoDecoder
from video_keyframe.scene_detector import HSVSceneDetector
from video_keyframe.keyframe_selector import KeyFrameSelector
from video_keyframe.gallery_generator import GalleryGenerator

# 1. 提取关键帧
selector = KeyFrameSelector(
    video_path="video.mp4",
    output_dir="keyframes",
    threshold=0.3,
    min_scene_length=1.0
)
keyframes = selector.select_keyframes()

# 2. 生成画廊
generator = GalleryGenerator("keyframes")
generator.generate(keyframes, selector.get_video_info(), "video.mp4")
```

## 性能优化建议

1. **调整采样间隔**: `--sample-interval` 越大处理越快，默认为2
2. **提高阈值**: `-t 0.5` 减少检测到的场景变化数量
3. **增加最小场景时长**: `-m 3.0` 避免提取过多关键帧

## 故障排除

### 检测不到足够的关键帧
- 降低阈值：`-t 0.2` 或更小
- 使用组合检测器：`--detector combined`

### 提取了太多相似帧
- 提高阈值：`-t 0.5`
- 增加最小场景时长：`-m 5.0`

## 许可证

MIT License
